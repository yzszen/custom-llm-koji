# BERTScore比較 + CSV保存

import json
import pandas as pd
from bert_score import score

# ファイルのパス
# before_file = "../data/koji_before_rag.jsonl"   # ベースモデル出力
# after_file  = "../data/koji_after_rag.jsonl"   # RAG後出力
before_file = "../data/koji_after_rag_0.jsonl"   # 初回AG後出力
after_file  = "../data/koji_after_rag_8.jsonl"   #  OFFICIAL_BOOST = 1.35 ⇒ 1.38
gold_file   = "../data/koji_eval_gold.jsonl"       # ゴールドラベル
output_csv  = "../data/csv/bert_score_results.csv"   # 保存先CSV

# --- JSONLファイル読み込み関数 ---
def load_jsonl(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# --- 各出力と参照（ゴールド）を読み込み ---
preds_before = load_jsonl(before_file)
preds_after  = load_jsonl(after_file)
references   = load_jsonl(gold_file)

# 各リストの output を抽出
before_outputs = [x["output"] for x in preds_before]  # not "outputs"
after_outputs  = [x["output"] for x in preds_after]
gold_outputs   = [x["output"] for x in references]

assert len(before_outputs) == len(after_outputs) == len(gold_outputs), "件数が一致しません"

# --- BERTScoreの計算（日本語指定）---
P_b, R_b, F1_b = score(before_outputs, gold_outputs, lang="ja", verbose=True)
P_a, R_a, F1_a = score(after_outputs, gold_outputs, lang="ja", verbose=True)

# --- データフレーム化 ---
rows = []
for i, (qa_b, qa_a, qa_gold, f1b, f1a) in enumerate(zip(preds_before, preds_after, references, F1_b, F1_a), 1):
    rows.append({
        "QID": i,
        "instruction": qa_b["instruction"],
        "input": qa_b.get("input", ""),
        "gold_output": qa_gold["output"],
        # "koji_base_output": qa_b["output"],  # 
        # "koji_rag_output": qa_a["output"],  # 
        "koji_rag_output": qa_b["output"],  # 
        "koji_rag_BOOST = 1.35 ⇒ 1.38_output": qa_a["output"],  #  OFFICIAL_BOOST = 1.35 ⇒ 1.38
        "F1_after": f1a.item(),
        "ΔF1": (f1a - f1b).item()
    })

df = pd.DataFrame(rows)

# --- CSV保存 ---
df.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"\n✅ BERTScore結果をCSVに保存しました: {output_csv}")
print("👉 Notionやスプレッドシートにそのまま読み込めます！")

# chatコードと照合済

# BERTScore.pyよりペースト 2025.9.9.19:02
# --- 結果表示 ---
print("\n✅ BERTScore(平均)")
print(f"Before - P: {P_b.mean():.4f}, R: {R_b.mean():.4f}, F1: {F1_b.mean():.4f}")
print(f"After  - P: {P_a.mean():.4f}, R: {R_a.mean():.4f}, F1: {F1_a.mean():.4f}")