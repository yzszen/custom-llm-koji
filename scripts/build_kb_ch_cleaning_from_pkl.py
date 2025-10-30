import os
import re
import pickle
import faiss
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

# =============================
# 設定
# =============================
KB_DIR = "../data/kb_merged"
KB_PKL = os.path.join(KB_DIR, "kb_chunks.pkl")
KB_META_PKL = os.path.join(KB_DIR, "kb_meta.pkl")
KB_INDEX = os.path.join(KB_DIR, "kb_chunks.index")

# 埋め込みモデル
# EMBED_MODEL = "sentence-transformers/stsb-xlm-r-multilingual"  # --- Emb_model unified(1/3) --- 下記に置換
EMBED_MODEL = "intfloat/multilingual-e5-base"
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer(EMBED_MODEL, device=device)

# キーワード（分野外フィルタ用）
# KEYWORDS = ["ギター", "弦", "コード", "奏法", "ピック", "アンプ"]
KEYWORDS = ["ギター", "弦", "コード", "奏法", "ピック", "アンプ", "フレット"]

# =============================
# 1. データ読み込み
# =============================
with open(KB_PKL, "rb") as f:
    chunks = pickle.load(f)
with open(KB_META_PKL, "rb") as f:
    metas = pickle.load(f)

print(f"Loaded {len(chunks)} chunks and {len(metas)} metas")

# =============================
# 2. 基本クリーニング
# =============================
def clean_text(text: str) -> str:
    text = re.sub(r"\[.*?\]", "", text)       # 脚注 [1], [要出典] の削除
    text = re.sub(r"<.*?>", "", text)         # HTMLタグ削除
    text = text.replace("　", " ")            # 全角スペースを半角に
    text = re.sub(r"\s+", " ", text).strip()  # 空白を整理
    return text

clean_chunks = [clean_text(t) for t in chunks]

# =============================
# 3. 短すぎる/ノイズ除去
# =============================
print("Filtering too short texts...")
filtered_chunks = []
filtered_metas = []
for t, m in zip(clean_chunks, metas):
    if len(t) >= 15 and not re.match(r"^[\-\・\=\s]+$", t):
        filtered_chunks.append(t)
        filtered_metas.append(m)

print(f"After length filter: {len(filtered_chunks)}")

# # =============================
# # 4. Embedding計算
# # =============================
# print("Encoding texts...")
# embeddings = embedder.encode(
#     filtered_chunks,
#     convert_to_tensor=True,
#     show_progress_bar=True,
#     # device="cuda" if embedder.device.type == "cuda" else "cpu"   # --- Emb_model unified(2/3) --- 下記ブロックに置換
# )

# =============================
# 4. 埋め込み計算（バッチ処理 + passage: プレフィックス）
# =============================
def encode_texts(texts: List[str], batch_size: int = 64) -> torch.Tensor:
    embeddings_list = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # passage: プレフィックスを付与
        batch_passage = [f"passage: {t}" for t in batch]
        # emb = embedder.encode(batch_passage, convert_to_tensor=True, show_progress_bar=False)  # --- Emb_model unified(2+追加1/3) --- 下記に置換
        emb = embedder.encode(batch_passage, convert_to_numpy=True, show_progress_bar=False)
        embeddings_list.append(emb)
    # return torch.cat(embeddings_list, dim=0)  # --- Emb_model unified(2+追加3/3) --- 下記に置換
    return np.vstack(embeddings_list)

print("Encoding embeddings...")
embeddings = encode_texts(filtered_chunks, batch_size=64)

# =============================
# 5. 重複削除（類似度 > 0.95）
# =============================
print("Removing near-duplicate entries...")
# cos_sim = cosine_similarity(embeddings.cpu().numpy())  # --- Emb_model unified(2+追加2/3) --- 下記に置換
cos_sim = cosine_similarity(embeddings)
np.fill_diagonal(cos_sim, 0.0)  # diagonal: 対角線

to_remove = set()
# threshold = 0.95
# threshold = 0.85  # --- try ---
threshold = 0.92  # --- try ---
for i in range(len(filtered_chunks)):
    if i in to_remove:
        continue
    dup_idx = np.where(cos_sim[i] > threshold)[0]
    for j in dup_idx:
        to_remove.add(j)

dedup_chunks = [t for i, t in enumerate(filtered_chunks) if i not in to_remove]
dedup_metas = [m for i, m in enumerate(filtered_metas) if i not in to_remove]

print(f"After duplicate removal: {len(dedup_chunks)}")  # duplicate: 重複

# =============================
# 6. 分野外フィルタ（任意）
# =============================
# def is_relevant(text):  # relevant: 関連する
#     return any(kw in text for kw in KEYWORDS)

# domain_chunks = []
# domain_metas = []
# for t, m in zip(dedup_chunks, dedup_metas):
#     if is_relevant(t):
#         domain_chunks.append(t)
#         domain_metas.append(m)

# print(f"After domain filtering: {len(domain_chunks)}")

domain_chunks = dedup_chunks
domain_metas = dedup_metas

# =============================
# 7. 保存
# =============================
os.makedirs(KB_DIR, exist_ok=True)

with open(KB_PKL, "wb") as f:
    pickle.dump(domain_chunks, f)
with open(KB_META_PKL, "wb") as f:
    pickle.dump(domain_metas, f)

print(f"Saved cleaned KB: {KB_PKL}, {KB_META_PKL}")

# =============================
# 8. FAISS Index 作成
# =============================
print("Encoding final embeddings for FAISS index...")
# final_embeddings = embedder.encode(domain_chunks, convert_to_numpy=True, show_progress_bar=True)  # --- Emb_model unified(3/3) --- 下記に置換
# final_embeddings = encode_texts(domain_chunks, batch_size=64).numpy() --- Emb_model unified(3+追加/3) --- 下記に置換
final_embeddings = encode_texts(domain_chunks, batch_size=64)
faiss.normalize_L2(final_embeddings)

index = faiss.IndexFlatIP(final_embeddings.shape[1])  # 内積ベース
index.add(final_embeddings)
faiss.write_index(index, KB_INDEX)
print(f"Saved FAISS index: {KB_INDEX}")

print("✅ Cleaning and indexing complete!")
