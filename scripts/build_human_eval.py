"""Build a blinded 20-query human-evaluation sheet and its separate answer key."""
import csv
import json
import os
from pathlib import Path
import random

from benchmarks.faithfulness import corrupt_summary

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs" / "benchmarks" / "faithfulness.json"
OUT = ROOT / "docs" / "benchmarks" / "human_eval_blind.csv"
KEY = ROOT / "docs" / "benchmarks" / "human_eval_key.csv"


def main():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    rng = random.Random(42)
    rows, key_rows = [], []
    for entry in artifact["results"]["queries"]:
        outputs = entry["outputs"]
        gemma = outputs[0]
        if len(outputs) > 1 and os.getenv("FIREWORKS_API_KEY"):
            other, other_label = outputs[1]["summary"], outputs[1]["label"]
        else:
            other, other_label = corrupt_summary(gemma["summary"], entry["query"])["unsupported"], "Gemma with deterministic corrupted sentence"
        candidates = [("Local Gemma", gemma["summary"]), (other_label, other)]
        rng.shuffle(candidates)
        abstracts = "\n\n".join(f"[{i + 1}] {paper['title']}\n{paper['abstract']}" for i, paper in enumerate(entry["papers"]))
        rows.append({"item_id": entry["id"], "query": entry["query"], "selected_abstracts": abstracts, "output_a": candidates[0][1], "output_b": candidates[1][1], "output_a_faithfulness_1_to_5": "", "output_b_faithfulness_1_to_5": "", "output_a_usefulness_1_to_5": "", "output_b_usefulness_1_to_5": "", "which_output_is_better_A_or_B_or_tie": "", "notes": ""})
        key_rows.append({"item_id": entry["id"], "output_a_system": candidates[0][0], "output_b_system": candidates[1][0]})
    with OUT.open("w", newline="", encoding="utf-8") as file: csv.DictWriter(file, fieldnames=rows[0].keys()).writeheader(); csv.DictWriter(file, fieldnames=rows[0].keys()).writerows(rows)
    with KEY.open("w", newline="", encoding="utf-8") as file: csv.DictWriter(file, fieldnames=key_rows[0].keys()).writeheader(); csv.DictWriter(file, fieldnames=key_rows[0].keys()).writerows(key_rows)
    print(f"Wrote blinded sheet {OUT} and separate key {KEY}")


if __name__ == "__main__": main()
