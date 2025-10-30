#!/usr/bin/env python3
# infer_with_rerank_boost.py
# infer_with_rerank.py をもとに修正（不要なコメントアウト等削除）
"""
RAG付き推論（ベースモデル: llm-jp など）＋FAISS初期取得＋Cross-Encoderによる再ランキング
変更点:
 - kb_meta.pkl を読み込み、公式サイトチャンクに対するスコア補正（倍率）を適用
 - プロンプトで公式サイト優先の指示を追加
 - E5 embedder で query に "query:" プレフィックスをつけてエンコード
"""

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

# ----------------------------
# 設定（必要に応じて変更）
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ベース生成モデル（ここを差し替えれば別モデルを使えます）
BASE_MODEL_ID = "llm-jp/llm-jp-3.1-1.8b-instruct4"

# 8bit 量子化設定（モデルに応じて不要なら削る）
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

# KB 関連ファイル
KB_DIR = "../data/kb_merged"
KB_CHUNKS_PKL = os.path.join(KB_DIR, "kb_chunks.pkl")
KB_META_PKL = os.path.join(KB_DIR, "kb_meta.pkl")  # --- meta_boost ---(1/10) 追加
KB_INDEX = os.path.join(KB_DIR, "kb_chunks.index")

# 埋め込み(E5で統一）) / reranker モデル
EMBED_MODEL = "intfloat/multilingual-e5-base"
# --- delete ---
CROSS_ENCODER_MODEL = "hotchpotch/japanese-reranker-cross-encoder-base-v1"

# retrieval / rerank パラメータ
INITIAL_TOP_K = 20     # FAISS で一旦引く候補数
# INITIAL_TOP_K = 50
RERANK_TOP_K = 5       # Cross-Encoder で上位何件を最終選出するか
# RERANK_TOP_K = 3
# ↑ 初期候補を多めにしてから rerank で絞るのが安定  # --- meta_boost ---(2/10) 次行を追加
# OFFICIAL_BOOST = 1.05  # is_official=True の場合に乗じる倍率（小さめのブースト）
OFFICIAL_BOOST = 1.40

# generation パラメータ
# MAX_NEW_TOKENS = 256
# MAX_NEW_TOKENS = 192
MAX_NEW_TOKENS = 384
DO_SAMPLE = False
REPETITION_PENALTY = 1.1

# system prompt：公式サイト優先を明示（retriever 側補正と併用）
system_prompt = """以下の質問に日本語で必ず3文以上で、事実に基づいて簡潔に回答してください。
回答作成時、参考情報として示した「公式サイト（メーカー・公式ページ等）」を優先して参照し、必要があれば出典（サイト名やURL）を明記してください。
"""
# system_prompt = """以下の質問に日本語で事実に基づいて回答してください。
# 2〜4文で、要点を整理して述べてください。
# 回答作成時、参考情報として示した「公式サイト（メーカー・公式ページ等）」を優先して参照し、必要があれば出典（サイト名やURL）を明記してください。
# """

# system_prompt = """以下の質問に日本語で必ず2文以上で、事実に基づいて簡潔に回答してください。
# 回答作成時、参考情報として示した「公式サイト（メーカー・公式ページ等）」を優先して参照し、必要があれば出典（サイト名やURL）を明記してください。
# """
# system_prompt = """以下の質問に日本語で3文以上で、事実に基づき簡潔かつ要点を絞って回答してください。
# 回答作成時、参考情報として示した「公式サイト（メーカー・公式ページ等）」を優先して参照し、必要があれば出典（サイト名やURL）を明記してください。
# """
# system_prompt = """以下の質問に日本語で必ず3文以上で、事実に基づいて回答してください。
# 回答作成時、参考情報として示した「公式サイト（メーカー・公式ページ等）」を優先して参照し、必要があれば出典（サイト名やURL）を明記してください。
# """

# system_prompt = """以下の質問に、事実に基づいて簡潔に回答してください。
# 回答作成時、参考情報として示した「公式サイト（メーカー・公式ページ等）」を優先して参照し、必要があれば出典（サイト名やURL）を明記してください。
# """
# system_prompt = """以下の質問に、事実に基づいて簡潔かつ明確に回答してください。
# 必要に応じて1〜3文程度でまとめてください。
# 参考情報として示した「公式サイト（メーカー・公式ページ等）」を優先して参照し、必要があれば出典（サイト名やURL）を明記してください。
# """
# system_prompt = """以下の質問に日本語で必ず3文以上で回答してください。  # --- meta_boost ---(3/10) 次行を追加
# """

# ----------------------------
# 1) ベース生成モデルロード
# ----------------------------
logging.info("Loading base causal LM (8bit/device_map=auto)...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"  # Automatic GPU/CPU allocation
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# generation device (入力テンソルを置く先)
use_cuda = torch.cuda.is_available()
gen_device = "cuda" if use_cuda else "cpu"
logging.info(f"Generation device: {gen_device}")

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

# ----------------------------
# 5) QA 推論ループ
# ----------------------------
# テスト用 QA リスト（ユーザーの既存ファイルに合わせる）
with open("../data/koji_eval_gold.jsonl", "r", encoding="utf-8") as f:
    qa_list = [json.loads(line) for line in f]

results = []
start_all = time.time()
# --- delete ---

for i, qa in enumerate(qa_list, 1):
    query = qa["input"]
    # ------------ rerank ------------↓
    # 1) retrieve + rerank (公式ブースト適用済み)
    # top_texts, top_indices, cross_scores = retrieve_and_rerank(query, initial_k=INITIAL_TOP_K, top_k=RERANK_TOP_K)  # --- meta_boost ---(8/10) 下記と続くブロックに置換
    top_texts, top_indices, top_candidates = retrieve_and_rerank(query, initial_k=INITIAL_TOP_K, top_k=RERANK_TOP_K)

    # # When building context, include top_texts joined; include citation markers optionally(オプションで引用マーカーを含める)
    # if top_texts:
    #     # include short citation tag to help model identify sources (optional) (モデルがソースを識別できるように短い引用タグを含める（オプション）
    #     context = ""
    #     for idx, txt in enumerate(top_texts, 1):
    #         # include index/id to help trace
    #         context += f"[参照{idx}] {txt}\n"
    # else:
    #     context = ""  # no retrieved context

    # build context: include citation markers and source info (prefer official)
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
    # ------------ rerank ------------↑

    # 2) prompt 作成（system_prompt に公式優先指示を追加済み）
    prompt = f"""{system_prompt}

参考情報:
{context}

質問: {query}
答え:"""

    # 3) generation inputs
    inputs = tokenizer(prompt, return_tensors="pt").to(gen_device)
    inputs.pop("token_type_ids", None)

    # 4) generate
    outputs_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=DO_SAMPLE, repetition_penalty=REPETITION_PENALTY)
    output_text = tokenizer.decode(outputs_ids[0], skip_special_tokens=True)
    # strip prompt echo if present
    if output_text.startswith(prompt):
        output_text = output_text[len(prompt):].strip()  #--- 2025.8.68追加 infer.ipynbより
    else:
        # best-effort: remove leading part up to last newline if prompt echo partially present(プロンプトエコーが部分的に存在する場合は、最後の改行までの先頭部分を削除)
        output_text = output_text.strip()

    # 5) collect retrieved_for_save using original metas for traceability
    retrieved_for_save = []
    for idx in top_indices:
        if 0 <= idx < len(kb_chunks):
            # save both text and meta (for debugging / audit)
            # retrieved_for_save.append(kb_chunks[idx])    # --- meta_boost ---(9/10) if-elseともに置換、results.append()は変更なし
            retrieved_for_save.append({
                "text": kb_chunks[idx],
                "meta": kb_metas[idx]
            })
        else:
            # retrieved_for_save.append("")
            retrieved_for_save.append({"text": "", "meta": {}})

    results.append({
        "instruction": qa.get("instruction", ""),
        "input": query,
        "retrieved": retrieved_for_save,
        "output": output_text
    })

    logging.info(f"[{i}/{len(qa_list)}] Done: retrieved={len(retrieved_for_save)}, out_len={len(output_text)}")

# save outputs # --- meta_boost --- (10/10) out_fileを変更
out_file = "../data/koji_after_rag.jsonl"
with open(out_file, "w", encoding="utf-8") as f:
    for obj in results:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

logging.info(f"RAG推論（rerank＋official-boost）完了! 保存: {out_file}  実行時間: {time.time()-start_all:.1f}s")

# git確認済（chunks, ES系, rerank, boost）