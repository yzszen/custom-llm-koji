import os
import re
import fitz  # PyMuPDF
import logging
from pathlib import Path
from urllib.parse import urlparse
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def download_pdf_from_url(url: str, dest_dir: Path) -> Path:
    """URL指定されたPDFをダウンロードして保存"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    file_name = os.path.basename(urlparse(url).path) or "downloaded.pdf"
    dest_path = dest_dir / file_name

    try:
        logging.info(f"Downloading PDF from {url}")
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(resp.content)
        logging.info(f"Saved: {dest_path}")
        return dest_path
    except Exception as e:
        logging.error(f"Failed to download {url}: {e}")
        return None


def clean_text_block(text: str) -> str:
    """段落補正と余分な改行の除去"""
    # 不要な制御文字除去
    text = re.sub(r"[\r\t]+", " ", text)
    # ハイフンによる単語分断を結合
    text = re.sub(r"-\n", "", text)
    # 段落マーカーを維持しつつ改行整理
    text = text.replace("\n\n", "§§")  # §§: 節（section）
    text = text.replace("\n", " ")
    text = text.replace("§§", "\n\n")
    # 多重スペース除去
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def pdf_to_txt(pdf_path: Path, output_dir: Path):
    """単一PDF → TXT変換"""
    try:
        logging.info(f"Processing: {pdf_path.name}")
        doc = fitz.open(pdf_path)
        text_blocks = []
        for page in doc:
            text = page.get_text("text")  # layoutではなくtext：自然な段落保持
            text_blocks.append(clean_text_block(text))
        full_text = "\n\n".join(text_blocks)

        # 出力ファイル保存
        output_dir.mkdir(parents=True, exist_ok=True)
        txt_path = output_dir / (pdf_path.stem + ".txt")
        txt_path.write_text(full_text, encoding="utf-8")
        logging.info(f"✅ Saved TXT: {txt_path}")
    except Exception as e:
        logging.error(f"Error converting {pdf_path.name}: {e}")


def convert_pdfs(input_source: str, output_dir: str = "../data/local_docs"):
    """
    PDF変換エントリポイント
    input_source:
        - ローカルPDFフォルダパス
        - 単一PDFファイルパス
        - PDFファイルURL（https://～.pdf）
    """
    input_path = Path(input_source)
    output_path = Path(output_dir)

    if input_source.lower().startswith("http"):
        pdf_path = download_pdf_from_url(input_source, Path("./downloads"))
        if pdf_path:
            pdf_to_txt(pdf_path, output_path)
        return

    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        pdf_to_txt(input_path, output_path)
    elif input_path.is_dir():
        pdf_files = sorted([f for f in input_path.glob("*.pdf")])
        if not pdf_files:
            logging.warning(f"No PDF files found in {input_path}")
            return
        for f in pdf_files:
            pdf_to_txt(f, output_path)
    else:
        logging.error(f"Invalid input: {input_source}")


if __name__ == "__main__":
    # === 使用例 ===
    # ローカルフォルダ内のPDFをすべてTXT化：
    # convert_pdfs("./pdfs")
    #
    # 単一PDF：
    # convert_pdfs("./sample.pdf")
    #
    # ネットURL：
    # convert_pdfs("https://example.com/report.pdf")

    convert_pdfs("./pdfs")  # デフォルト例

