# build_kb_ch_boost_multi_cat.py
# build_kb_ch_boost.py をベースに変更（不要なコメント行及び更新履歴はここで削除）
#!/usr/bin/env python3

import os
import re
import time
import pickle
import logging
import math  # --- 均等カテゴリ対応
import torch  # --- GPU対応に（均等サンプリング以降）
from typing import List, Dict, Set

import faiss
import numpy as np
import wikipediaapi
from sentence_transformers import SentenceTransformer
import requests  # --- category_hp_merged --- ここから3行追加
from bs4 import BeautifulSoup
import json

# ----------------------------
# 設定
# ----------------------------
SAVE_DIR = "../data/kb_merged"
os.makedirs(SAVE_DIR, exist_ok=True)

KB_PKL = os.path.join(SAVE_DIR, "kb_chunks.pkl")
KB_META_PKL = os.path.join(SAVE_DIR, "kb_meta.pkl")
KB_INDEX = os.path.join(SAVE_DIR, "kb_chunks.index")

USER_AGENT = "llmjp_kb_builder/1.0"
WIKI_LANG = "ja"

# チャンク設定
CHUNK_SIZE = 400       # チャンク長（文字）。環境に応じて 200-700 を検討
OVERLAP = 80           # オーバーラップ（文字）
# CHUNK_SIZE = 500
# OVERLAP = 100
MIN_CHUNK_LEN = 30     # 無視する短いチャンク長

# カテゴリ設定
# CATEGORY_NAME = "Category:ギター"   # 起点にするカテゴリ名（変更可）  # --- multi_cat ---(1/4) 次行追加
CATEGORY_NAMES = [
    "Category:発酵",
    "Category:発酵食品"
]
# MAX_PAGES = 500        # 取得上限ページ数（安全用）# None にすると全件収集
MAX_PAGES = None
MAX_SUBCATEGORY_DEPTH = 1  # サブカテゴリを辿る深さ（0 = 直下のみ, 1 = 直下+その子）

# 埋め込みモデル
EMBED_MODEL = "intfloat/multilingual-e5-base"

# ロギング
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# 公式サイトリスト
HP_URLS_FILE = "../data/hp_urls.json"

# ローカル文書ディレクトリ
LOCAL_DOCS_DIR = "../data/local_docs"  # <-- ここに自作 .txt を入れる

# ----------------------------
# Utility
# ----------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_into_chunks(text: str, chunk_size=CHUNK_SIZE, overlap=OVERLAP) -> List[str]:
    """単純な文字ベースのチャンク分割（オーバーラップあり）"""
    chunks = []
    text = text.strip()
    if len(text) <= chunk_size:
    #     if len(text) >= MIN_CHUNK_LEN:  # --- Local Markdown support ---(1/4)の際下記1行に
    #         return [text]
    #     return []
        return [text] if len(text) >= MIN_CHUNK_LEN else []
    start = 0
    L = len(text)
    while start < L:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHUNK_LEN:
            chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks


# ============================================================
# Markdown構造対応版（ローカル文書専用）
# ============================================================# --- Local Markdown support ---(2/4)新規ブロック（心臓部分）
def split_into_structured_chunks(text, max_len=1000):
    """
    Markdown構造を考慮してチャンク化する
    - #, ## などの見出しを境に分割
    - 空行や箇条書きを保持
    """
    # sections = re.split(r"(?=^#+\s)", text, flags=re.MULTILINE)  # ---【## セクションの先頭に、冒頭の # タイトル文字列をプレフィックスとして付ける】修正(1/4) 次行に置換
    sections = re.split(r"(?=^##\s)", text, flags=re.MULTILINE)
    chunks = []

    # 最初のタイトル（# ...）を抽出  # ---【## セクションの先頭に、冒頭の # タイトル文字列をプレフィックスとして付ける】修正(2/4) このブロック（3行）を追加
    title_match = re.search(r"^#\s*(.+)", sections[0])
    global_title = title_match.group(0).strip() if title_match else ""

    for sec in sections:
        sec = sec.strip()
        # if not sec:  # ---【## セクションの先頭に、冒頭の # タイトル文字列をプレフィックスとして付ける】修正(3/4) 次行に置換
        if not sec or sec.startswith("# "):  # タイトル単独ブロックはスキップ
            continue


        # 各チャンクの冒頭にタイトルを追加  # ---【## セクションの先頭に、冒頭の # タイトル文字列をプレフィックスとして付ける】修正(4/4) このブロック（3行）を追加
        if global_title:
            sec = f"{global_title}\n\n{sec}"

        paragraphs = sec.split("\n\n")
        current_chunk = ""
        for p in paragraphs:
            if len(current_chunk) + len(p) < max_len:
                current_chunk += p.strip() + "\n\n"
            else:
                chunks.append(current_chunk.strip())
                current_chunk = p.strip() + "\n\n"
        if current_chunk:
            chunks.append(current_chunk.strip())

    return chunks


# ----------------------------
# Wikipedia crawl
# ----------------------------
wiki = wikipediaapi.Wikipedia(language=WIKI_LANG, user_agent=USER_AGENT)

# def collect_category_pages(category_name: str, max_depth: int = 1, max_pages: int = 500) -> List[wikipediaapi.WikipediaPage]:  # --- category_hp_merged --- 次行に差替
def collect_category_pages(category_name: str, max_depth: int = 1, max_pages: int = 500):
    """
    カテゴリ内のページを収集（サブカテゴリを深さ max_depth まで辿る）
    # 戻り値は wikipediaapi の page オブジェクトのリスト（記事ページのみ）
    均等サンプリング対応:
      - max_pages=None の場合: 全件収集
      - max_pages が指定されている場合: 全件から均等にサンプリング
    """
    logging.info(f"Collect pages from category: {category_name} depth={max_depth} max_pages={max_pages}")
    cat = wiki.page(category_name)
    if not cat.exists():
        logging.error(f"Category not found: {category_name}")
        return []

    all_pages = []
    visited_cats: Set[str] = set()

    def recurse(cat_page, depth):
        if cat_page.title in visited_cats:
            return
        visited_cats.add(cat_page.title)
        # categorymembers returns dict of title -> page
        members = cat_page.categorymembers
        for _, member in members.items():
            # member.ns == 0 -> article/page; ns == 14 -> category
            if member.ns == wikipediaapi.Namespace.MAIN:
                all_pages.append(member)
            elif member.ns == wikipediaapi.Namespace.CATEGORY and depth > 0:
                # Explore subcategories
                recurse(member, depth - 1)

    recurse(cat, max_depth)
    logging.info(f"Collected {len(all_pages)} total pages before sampling")

    # --- 均等サンプリング処理 ---
    if max_pages is None or max_pages >= len(all_pages):
        return all_pages
    else:
        step = len(all_pages) / max_pages
        sampled = [all_pages[math.floor(i * step)] for i in range(max_pages)]
        logging.info(f"Sampled {len(sampled)} pages (均等サンプリング)")
        return sampled

# page -> chunk generation
def page_to_chunks(page: wikipediaapi.WikipediaPage) -> List[Dict]:
    """
    ページ本文を段落ごとに分けてチャンク化し、チャンクのメタを返す。
    返却リストの要素は dict:
      {"text": chunk_text, "source_title": page.title, "source_url": page.fullurl}
    """
    chunks = []
    text = page.text or ""
    if not text:
        return chunks
    # 段落分割（改行で分ける）  # --- S2C(SECTION TO CHUNK) ---(1/6) 次行追加
    pos = 0
    for para in text.split("\n"):
        p = clean_text(para)
        if len(p) >= MIN_CHUNK_LEN:
            chs = split_into_chunks(p, CHUNK_SIZE, OVERLAP)
            for ch in chs:
                chunks.append({
                    "text": ch,
                    "source_title": page.title,
                    "source_url": page.fullurl,
                    "chunk_id": None,   # 後で付与  # --- S2C ---(2/6) ここから３行+ pos += 1 追加
                    "position": pos,
                    "is_official": False
                })
                pos += 1
    return chunks

# ----------------------------
# Official site crawl  #  200: OK TCP_p320
# ----------------------------
def scrape_official_site(url: str) -> List[Dict]:
    try:
        headers = {"User-Agent": USER_AGENT}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            logging.warning(f"Failed to fetch {url}, status {r.status_code}")
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        texts = []
        for p in soup.find_all(["p", "div", "li"]):
            t = clean_text(p.get_text())
            if len(t) >= MIN_CHUNK_LEN:
                texts.append(t)
        chunks = []
        pos = 0
        for t in texts:
            chs = split_into_chunks(t, CHUNK_SIZE, OVERLAP)
            for ch in chs:
                chunks.append({
                    "text": ch,
                    "source_title": "official_site",
                    "source_url": url,
                    "chunk_id": None,  # --- S2C ---(3/6) ここから３行 + ９行前のpos = 0 + pos += 1 追加
                    "position": pos,
                    "is_official": True
                })
                pos += 1
        return chunks
    except Exception as e:
        logging.warning(f"Error scraping {url}: {e}")
        return []

def collect_official_sites(hp_file: str) -> List[Dict]:
    """JSONファイルに書かれたURLリストを読み込み、対象URLをscrape_official_site()でスクレイピングしてテキストチャンクを収集し、ひとつのリストとして返す"""
    if not os.path.exists(hp_file):
        logging.warning(f"No hp_url file found: {hp_file}")
        return []
    with open(hp_file, "r") as f:
        urls = json.load(f)
    all_chunks = []
    for url in urls:
        all_chunks.extend(scrape_official_site(url))
    logging.info(f"Collected {len(all_chunks)} chunks from official sites")
    return all_chunks

# ----------------------------
# Local docs reader (自作データを直接KBに追加): 構造化チャンク版
# ----------------------------
def load_local_docs(local_dir: str) -> List[Dict]:
    """ローカル .txt ファイルを読み込み、チャンク化して返す"""
    if not os.path.exists(local_dir):
        logging.warning(f"No local docs dir found: {local_dir}")
        return []
    all_chunks = []
    # pos = 0
    # for fname in os.listdir(doc_dir):
    #     if not fname.endswith((".md", ".txt")):
    #         continue
    #     fpath = os.path.join(doc_dir, fname)
    txt_files = [f for f in os.listdir(local_dir) if f.endswith(".txt")]
    for fname in txt_files:
        fpath = os.path.join(local_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read()
            # text = clean_text(text)  # --- 対策 >> collected 0 chunks from local docs (4 files) ---

            # if len(text) < MIN_CHUNK_LEN:  # コードは適切だがコメントアウトで回避する
            #     continue

            # チャンク分割
            # pos = 0  # --- Local Markdown support ---(3/4) 次ブロックに内包 
            # chs = split_into_chunks(text, CHUNK_SIZE, OVERLAP)
            # for ch in chs:
            # for ch in split_into_chunks(text, CHUNK_SIZE, OVERLAP):  # --- Local Markdown support ---(4/4) 次２行に置換 
            # ✅ 構造化チャンク化を適用
            for pos, ch in enumerate(split_into_structured_chunks(text, max_len=1000)):
                all_chunks.append({
                    "text": ch,
                    "source_title": fname,
                    # "source_url": f"file://{fpath}",  # URL的に扱えるように
                    "source_url": f"local://{fname}",
                    "chunk_id": None,
                    "position": pos,
                    "is_official": True  # 明示的にローカルを「公式扱い」にしてもよい
                })
                # pos += 1  # --- Local Markdown support ---(3.5/4) 上記ブロックに内包
        except Exception as e:
            logging.warning(f"Error reading {fname}: {e}")
    # logging.info(f"Loaded {len(chunks)} chunks from local docs in {doc_dir}")
    # return chunks
    logging.info(f"Collected {len(all_chunks)} chunks from local docs ({len(txt_files)} files)")
    return all_chunks

# ----------------------------
# Build KB
# ----------------------------
# --- delete ---
def build_kb(save_dir: str = SAVE_DIR):
    existing_texts = []
    existing_metas = None
    if os.path.exists(KB_PKL):
        try:
            with open(KB_PKL, "rb") as f:
                existing_texts = pickle.load(f)
            logging.info(f"Loaded existing KB chunks (len={len(existing_texts)}) from {KB_PKL}")
        except Exception as e:
            logging.warning(f"Failed to load existing KB chunks: {e}")
            existing_texts = []

    # try to load existing meta file if present (preserve original metadata)
    if os.path.exists(KB_META_PKL):
        try:
            with open(KB_META_PKL, "rb") as f:
                existing_metas = pickle.load(f)
            logging.info(f"Loaded existing KB metas (len={len(existing_metas)}) from {KB_META_PKL}")
            # If both present, prefer metas: build merged_meta from metas directly
            if isinstance(existing_metas, list) and len(existing_metas) == len(existing_texts):
                # reconstruct merged_meta from existing_metas + texts
                merged_meta = []
                for t, m in zip(existing_texts, existing_metas):
                    # ensure all expected keys present
                    meta = {
                        "text": t,
                        "source_title": m.get("source_title", "existing"),
                        "source_url": m.get("source_url", ""),
                        "chunk_id": m.get("chunk_id", None),
                        "position": m.get("position", -1),
                        "is_official": m.get("is_official", False)
                    }
                    merged_meta.append(meta)
                logging.info("Reconstructed merged_meta from existing metas.")
            else:
                # existing_metas inconsistent with texts -> fallback to building merged_meta from texts
                merged_meta = [{"text": t, "source_title": "existing", "source_url": "", "chunk_id": None, "position": -1, "is_official": False} for t in existing_texts]
                logging.warning("existing meta len mismatch with texts; fallback to default metas.")
        except Exception as e:
            logging.warning(f"Failed to load existing KB metas: {e}")
            merged_meta = [{"text": t, "source_title": "existing", "source_url": "", "chunk_id": None, "position": -1, "is_official": False} for t in existing_texts]
    else:
        # no meta file: create default metas for existing texts
        merged_meta = [{"text": t, "source_title": "existing", "source_url": "", "chunk_id": None, "position": -1, "is_official": False} for t in existing_texts]
        logging.info("No existing meta file found; created default metas.")

    # 1) Wikipedia pages（複数カテゴリ対応）  # --- multi_cat ---(2/4) 次行を新ブロックで置換
    # pages = collect_category_pages(CATEGORY_NAME, MAX_SUBCATEGORY_DEPTH, MAX_PAGES)
    wiki_chunks_meta = []
    # for i, page in enumerate(pages, 1):  # 下記は # - のみ変更し、catループにこのブロックを入れ子している
    #     try:
    #         # --- delete ---
    #         wiki_chunks_meta.extend(page_to_chunks(page))
    #         if i % 50 == 0:
    #             logging.info(f"Processed {i}/{len(pages)} pages")    # -
    #     except Exception as e:
    #         logging.warning(f"Failed processing page {page.title}: {e}")
    # # --- delete ---
    # logging.info(f"Generated {len(wiki_chunks_meta)} chunks from Wikipedia")  # -

    for cat in CATEGORY_NAMES:
        pages = collect_category_pages(cat, MAX_SUBCATEGORY_DEPTH, MAX_PAGES)
        for i, page in enumerate(pages, 1):
            try:
                wiki_chunks_meta.extend(page_to_chunks(page))
                if i % 50 == 0:
                    logging.info(f"Processed {i}/{len(pages)} pages in {cat}")
            except Exception as e:
                logging.warning(f"Failed processing page {page.title}: {e}")
        logging.info(f"Generated {len(wiki_chunks_meta)} chunks from Wikipedia (so far)")

    # 2) 公式サイト
    official_chunks_meta = collect_official_sites(HP_URLS_FILE)

    # 2.5) Local docs
    # LOCAL_DOC_DIR = "../data/local_docs"
    local_chunks_meta = load_local_docs(LOCAL_DOCS_DIR)

    # 3) マージ（既存 + new） — 重複判定: テキストベース
    # new_chunks_meta = wiki_chunks_meta + official_chunks_meta
    new_chunks_meta = wiki_chunks_meta + official_chunks_meta + local_chunks_meta

    # Build a map for quick lookup from text -> index in merged_meta
    text_to_index = {m["text"]: idx for idx, m in enumerate(merged_meta)}

    added = 0
    updated_official_to_true = 0
    for meta in new_chunks_meta:
        t = meta["text"]
        if t in text_to_index:
            # duplicate text: consider updating is_official if new meta says True
            idx = text_to_index[t]
            existing = merged_meta[idx]
            if meta.get("is_official", False) and not existing.get("is_official", False):
                # promote to official (we want official to be preserved)
                existing["is_official"] = True
                # also prefer to keep source_title/source_url from official meta
                existing["source_title"] = meta.get("source_title", existing.get("source_title", ""))
                existing["source_url"] = meta.get("source_url", existing.get("source_url", ""))
                updated_official_to_true += 1
            # else: keep existing (don't overwrite other fields)
        else:
            # new unique text: append as-is (ensure keys present)
            merged_meta.append({
                "text": meta.get("text", ""),
                "source_title": meta.get("source_title", "") or "unknown",
                "source_url": meta.get("source_url", "") or "",
                "chunk_id": None,
                "position": meta.get("position", -1),
                "is_official": bool(meta.get("is_official", False))
            })
            text_to_index[t] = len(merged_meta) - 1
            added += 1

    # logging.info(f"Merged KB: existing_texts={len(existing_texts)} -> total merged_meta={len(merged_meta)} (added {added}, promoted official {updated_official_to_true})")  # --- multi_cat ---(3/4) 次行に置換
    logging.info(f"Merged KB -> total {len(merged_meta)} (added {added}, promoted official {updated_official_to_true})")

    # 4) chunk_id 付与（再割当）  # ---S2C ---(5/6) この３行追加
    for idx, m in enumerate(merged_meta):
        m["chunk_id"] = idx

    #テキストとメタを分離（保存形式）
    texts = [m["text"] for m in merged_meta]
    # metas = [{"source_title": m.get("source_title", ""), "source_url": m.get("source_url", "")} for m in merged_meta]  # ---S2C ---(6/6) この１行追加  
    metas = [{k: v for k, v in m.items() if k != "text"} for m in merged_meta]

    # 5) 埋め込み（passage: プレフィックス）
    logging.info(f"Loading embedder: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Computing embeddings...")
    # E5 では "passage:" プレフィックスを付けるのが良いことがあるので任意で付与可能  # --- multi_cat ---(4/4) 次行のpassage付与省略されているが残す。また続く embeddings = embedder.encode で batch_size=64, normalize_embeddings=True が追加されているが、後続との関係性を考慮し採用しない。
    passages = [f"passage: {t}" for t in texts]
    embeddings = embedder.encode(passages, convert_to_numpy=True, show_progress_bar=True)

    # 6) FAISS index 作成（cosine用正規化）
    faiss.normalize_L2(embeddings)
    # FAISS IndexFlatIP 作成
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logging.info(f"FAISS index built, ntotal={index.ntotal}, dim={dim}")

    # 7) 保存（chunks + metas + index）
    # --- delete --
    with open(KB_PKL, "wb") as f:
        # 保存形式を dict list （text + meta）にしたい場合はメタも一緒に保存するが
        # 後続コード互換のため `kb_chunks.pkl` を text list とするか、meta別保存にするか選べます。
        # ここでは chunks と meta を別ファイルで保存します（互換性を保つため両方）
        pickle.dump(texts, f)
    with open(KB_META_PKL, "wb") as f:
        pickle.dump(metas, f)
    faiss.write_index(index, KB_INDEX)

    # debug counts  # --- ここから③行追加 ---
    official_count = sum(1 for m in metas if m.get("is_official"))
    logging.info(f"Saved KB: texts={len(texts)}, metas={len(metas)}, official_count={official_count}")
    logging.info("KB build finished")
    return len(texts)

# ----------------------------
# 実行ブロック
# ----------------------------
if __name__ == "__main__":
    # logging.info("Start building KB from Wikipedia + official sites")
    logging.info("Start building KB from Wikipedia + official sites + local docs")
    total = build_kb(SAVE_DIR)
    logging.info(f"Finished. Total chunks saved: {total}")

