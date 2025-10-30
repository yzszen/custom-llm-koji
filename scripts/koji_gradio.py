# llmjp_lora_tuning.ipynb(project name: llmjp_lora_gardening) の gradioコードを参考に作成

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import faiss
import pickle
import json
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
import os
import time
import logging
from typing import List, Tuple

import csv  # flagで追加
from datetime import datetime

# ----------------------------
# 設定（必要に応じて変更）
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

BASE_MODEL_ID = "llm-jp/llm-jp-3.1-1.8b-instruct4"
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

KB_DIR = "../data/kb_merged"
KB_CHUNKS_PKL = os.path.join(KB_DIR, "kb_chunks.pkl")
KB_META_PKL = os.path.join(KB_DIR, "kb_meta.pkl")
KB_INDEX = os.path.join(KB_DIR, "kb_chunks.index")

EMBED_MODEL = "intfloat/multilingual-e5-base"
CROSS_ENCODER_MODEL = "hotchpotch/japanese-reranker-cross-encoder-base-v1"

INITIAL_TOP_K = 20
RERANK_TOP_K = 5
# OFFICIAL_BOOST = 1.05
OFFICIAL_BOOST = 1.40

# MAX_NEW_TOKENS = 256
MAX_NEW_TOKENS = 384
DO_SAMPLE = False
REPETITION_PENALTY = 1.1
system_prompt = """以下の質問に日本語で必ず3文以上で、事実に基づいて簡潔に回答してください。
回答作成時、参考情報として示した「公式サイト（メーカー・公式ページ等）」を優先して参照し、必要があれば出典（サイト名やURL）を明記してください。
"""

# ----------------------------
# 1) ベース生成モデルロード
# ----------------------------
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

use_cuda = torch.cuda.is_available()
gen_device = "cuda" if use_cuda else "cpu"
logging.info(f"Generation device: {gen_device}")


# -------------------------------------------------
# ----------------------------
# 2) KBロード（chunks + metas + FAISS index） + embedder
# ----------------------------
logging.info("Loading KB chunks and metas...")
with open(KB_CHUNKS_PKL, "rb") as f:
    kb_chunks = pickle.load(f)   # list of strings  # --- meta_boost ---(4/10) 次の2行を追加
with open(KB_META_PKL, "rb") as f:
    kb_metas = pickle.load(f)    # list of dicts (aligned with kb_chunks)

index = faiss.read_index(KB_INDEX)
logging.info(f"Loaded {len(kb_chunks)} chunks, FAISS ntotal={index.ntotal}")

logging.info("Loading embedder for query encoding...")
embedder = SentenceTransformer(EMBED_MODEL, device="cuda" if use_cuda else "cpu")

# ----------------------------
# 3) Cross-Encoderロード（reranker）
# ----------------------------
logging.info(f"Loading Cross-Encoder for reranking: {CROSS_ENCODER_MODEL}")
cross_device = "cuda" if use_cuda else "cpu"
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device=cross_device)

# ----------------------------
# 4) Retrieval + Rerank 関数群
# ---------------------------- # --- meta_boost --- では定義文のtypingやdocstring(Be_pp198)を手入れ。プレフィックスは対応済のため修正なし。
def retrieve_candidates(query: str, initial_k: int = INITIAL_TOP_K) -> Tuple[List[dict], List[int], List[float]]:
    """
    FAISS で initial_k 件を取り、(candidates_meta_list, indices, faiss_scores) を返す。
    candidates_meta_list: [{"text":..., "meta": {...}, "faiss_score": ...}, ...]
    """
    # embed query (with prefix)
    q_vec = embedder.encode([f"query: {query}"], convert_to_numpy=True)
    faiss.normalize_L2(q_vec)  # cosine 類似度用
    D, I = index.search(q_vec, initial_k)
    # ------------ rerank ------------↓
    # D: similarities, I: indices
    indices = [int(i) for i in I[0] if i != -1]
    faiss_scores = [float(s) for s in D[0][:len(indices)]]
    # candidates = [kb_chunks[i] for i in indices]  # --- meta_boost ---(5/10) 以下のブロックに置換
    candidates = []
    for idx, score in zip(indices, faiss_scores):
        meta = kb_metas[idx] if 0 <= idx < len(kb_metas) else {}
        text = kb_chunks[idx] if 0 <= idx < len(kb_chunks) else ""
        candidates.append({"text": text, "meta": meta, "faiss_score": score})
    # ------------ rerank ------------↑
    return candidates, indices, faiss_scores

def rerank_with_crossencoder(query: str, candidates: List[dict], top_k: int = RERANK_TOP_K) -> Tuple[List[dict], List[int], List[float]]:
    """
    Cross-Encoder で (query, passage) をスコア化し、公式サイトブーストを適用して top_k を返す。
    戻り値:
      top_candidates: list of candidate dicts (same structure, with added 'cross_score' and 'final_score')
      top_order_indices: original order indices into candidates
      final_scores: numpy-like list of final scores
    """
    if not candidates:
        return [], [], []

    # prepare pairs for cross-encoder
    # pairs = [(query, c) for c in candidates]  # --- meta_boost ---(6/10) 定義文のtypingやdocstring(Be_pp198)を手入れ + 下記に置換 + final_scores = [] & top_candidates = [] の各ブロック追加
    pairs = [(query, c["text"]) for c in candidates]
    cross_scores = cross_encoder.predict(pairs)  # numpy array
    # apply official boost (multiplier) to final score
    final_scores = []
    for c, cs in zip(candidates, cross_scores):
        is_official = bool(c.get("meta", {}).get("is_official", False))
        # combine cross encoder score and (optionally) faiss score if desired
        # Here we primarily use cross_score and then apply official multiplier
        final = float(cs) * (OFFICIAL_BOOST if is_official else 1.0)
        final_scores.append(final)

    # order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True) 
    # top_order = order[:top_k]
    # top_texts = [candidates[i] for i in top_order]
    # return top_texts, top_order, scores

    # sort by final_scores desc
    order = sorted(range(len(final_scores)), key=lambda i: final_scores[i], reverse=True)
    top_order = order[:top_k]
    top_candidates = []
    for i in top_order:
        cand = candidates[i].copy()
        cand["cross_score"] = float(cross_scores[i])
        cand["final_score"] = float(final_scores[i])
        top_candidates.append(cand)
    return top_candidates, top_order, final_scores

def retrieve_and_rerank(query: str, initial_k: int = INITIAL_TOP_K, top_k: int = RERANK_TOP_K):
    """
    高レベル: 1) FAISS: initial_k 候補取得 2) Cross-Encoder: 再ランキング 3) 公式ブースト適用 -> top_k を返却
    戻り値: top_texts (list of strings), top_indices (list of original global indices), top_candidates (detailed)
    """
    candidates, indices, faiss_scores = retrieve_candidates(query, initial_k)
    if not candidates:
        return [], [], []
    # top_texts, top_order, cross_scores = rerank_with_crossencoder(query, candidates, top_k=top_k)  # --- meta_boost ---(7/10) 下記ブロックに置換
    # # for downstream traceability, we might want to return the original indices of selected candidates
    # # 下流の追跡可能性のために、選択された候補の元のインデックスを返す必要があるかもしれません
    # top_indices = [indices[i] for i in top_order]
    # return top_texts, top_indices, cross_scores

    top_candidates, top_order_local, _ = rerank_with_crossencoder(query, candidates, top_k=top_k)
    # convert local order to original global indices
    top_indices = [indices[i] for i in top_order_local]
    top_texts = [c["text"] for c in top_candidates]
    return top_texts, top_indices, top_candidates


# ----------------------------------------------------


def answer_question(question):

    # --------------------------
    query = question
    top_texts, top_indices, top_candidates = retrieve_and_rerank(query, initial_k=INITIAL_TOP_K, top_k=RERANK_TOP_K)

    if top_candidates:
        context_lines = []
        for rank, cand in enumerate(top_candidates, 1):
            meta = cand.get("meta", {})
            src = meta.get("source_title", "") or meta.get("source_url", "")
            is_official = meta.get("is_official", False)
            # short indicator to help LM (we already boosted scores)
            tag = "（公式）" if is_official else ""
            context_lines.append(f"[参照{rank}{tag}] {cand['text']}")
        context = "\n".join(context_lines)
    else:
        context = ""

    # prompt = f"質問に答えてください\n{question}\n答え"
    prompt = f"""{system_prompt}

参考情報:
{context}

質問: {query}
答え:"""
    
    # --------------------------


    inputs = tokenizer(prompt, return_tensors="pt").to(gen_device)
    inputs.pop("token_type_ids", None)
    output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=DO_SAMPLE, repetition_penalty=REPETITION_PENALTY)

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)[len(prompt):].strip()

# demo = gr.Interface(
#     fn=answer_question,
#     # inputs =gr.Textbox(label="質問を入力"),
#     # outputs=gr.Textbox(label="モデルの回答"),
#     # title="QAデモ"
#     inputs =gr.Textbox(label="質問を入力", lines=5),
#     outputs=gr.Textbox(label="回答", lines=15),
#     title="法華経QA"
# )

# 縦表示
def clear_input():
    return ""

# def flag_output():
#     return "✔️ 回答はフラグされました。"

def flag_output(question, answer):
    os.makedirs(".gradio/flagged", exist_ok=True)
    timestamp = datetime.now().isoformat(sep=' ', timespec='seconds')  # flagで追加

    with open(".gradio/flagged/flagged.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # writer.writerow([question, answer])
        writer.writerow([question, answer, timestamp])  # flagで追加
    return "✔️ 回答はフラグされました。"




custom_css = """
#title {
    text-align: center;
    font-size: 2rem;
    padding: 1rem;
    # background-image: url('https://images.unsplash.com/photo-1464802686167-b939a6910659?q=80&w=1450&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D');
    background-size: cover;
    background-position: center;
    color: white;
    border-radius: 8px;
}

#submit-btn {
    # background-color: orange !important;
    # background-color: green !important;
    # background-color: #004d40 !important;
    background-color: lightcoral !important;
    color: white !important;
    font-weight: bold;
}
"""

with gr.Blocks(css=custom_css) as demo:  # css定義付与
    gr.Markdown("## 麹の世界", elem_id="title")  # タイトル中央＆背景 # elem_id付与

    with gr.Column():
        question_box = gr.Textbox(label="質問を入力", lines=5)

        with gr.Row():
            submit_btn = gr.Button("送信", elem_id="submit-btn") # elem_id付与
            clear_btn = gr.Button("Clear")  # Clearボタン追加

        # answr_box = gr.Textbox(label="回答", lines=15)
        answer_box = gr.Textbox(
            label="回答",
            lines=10,
            max_lines=15,
            autoscroll=True,  # ✅ 自動スクロール（長文でも軽い）
            interactive=False,
            show_copy_button=True,  # flag の際にこちらも追加
            # show_flag=True  # ✅ これで .gradio/flagged に保存される
        )
        flag_btn = gr.Button("Flag")  # Flagボタン追加

    submit_btn.click(fn=answer_question, inputs=question_box, outputs=answer_box)
    clear_btn.click(fn=clear_input, outputs=question_box)
    # flag_btn.click(fn=flag_output, outputs=answr_box)
    flag_btn.click(fn=flag_output, inputs=[question_box, answer_box], outputs=answer_box)
    

# demo.launch()  # # => gradioアップデートして実行したらつながった！！！(WSL側のexport GRADIO_SERVER_NAME=0.0.0.0 をログオフしてもOK) ただし、http://127.0.0.1:7860/ で表示され http://172.17.255.157:7860/ では表示されない
# demo.launch(server_name="0.0.0.0", server_port=7860)  # => gradioアップデートして実行したらつながった！！！(WSL側のexport GRADIO_SERVER_NAME=0.0.0.0 をログオフしてもOK) http://127.0.0.1:7860/ と http://172.17.255.157:7860/ のどちらでも表示される！！！
demo.launch(share=True)
# demo.launch(server_name="0.0.0.0", server_port=7860, root_path="http://172.17.255.157:7860/")  # 真っ白のままだった => gradioアップデートして実行したらつながった！！！
# demo.launch(server_name="0.0.0.0", server_port=7860, root_path="http://10.255.255.254:7860/")  # 応答時間が長すぎます(でアクセスできない) cat /etc/resolv.conf | grep nameserver （WSL2内で以下を実行してWindows側のIPを取得）

# 以下は経過
# demo.launch(GRADIO_SERVER_NAME="0.0.0.0", GRADIO_SERVER_PORT=7860)  # 試しに -> TypeError: Blocks.launch() got an unexpected keyword argument 'GRADIO_SERVER_NAME'
# demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=False, show_error=True, debug=True)
# demo.launch(server_name="*", server_port=7860, inbrowser=False, show_error=True, debug=True)
