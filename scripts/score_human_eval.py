"""Score a completed blinded evaluation sheet using its private answer key."""
import argparse, csv, json, random
from pathlib import Path

def mean(values): return sum(values) / len(values) if values else None
def interval(values, seed=42, rounds=5000):
    if not values: return None
    rng=random.Random(seed); samples=sorted(mean([rng.choice(values) for _ in values]) for _ in range(rounds))
    return [samples[int(.025*(rounds-1))], samples[int(.975*(rounds-1))]]
def main():
    p=argparse.ArgumentParser(); p.add_argument("sheet", type=Path); p.add_argument("--key", type=Path, default=Path("docs/benchmarks/human_eval_key.csv")); p.add_argument("--output", type=Path, default=Path("docs/benchmarks/human_eval_scores.json")); a=p.parse_args()
    keys={r["item_id"]:r for r in csv.DictReader(a.key.open(encoding="utf-8"))}; values={}
    wins={}
    for r in csv.DictReader(a.sheet.open(encoding="utf-8")):
        k=keys[r["item_id"]]
        for side in ("a","b"):
            system=k[f"output_{side}_system"]; bucket=values.setdefault(system,{"faithfulness":[],"usefulness":[]})
            for metric in bucket:
                value=r.get(f"output_{side}_{metric}_1_to_5", "").strip()
                if value: bucket[metric].append(float(value))
        better=r.get("which_output_is_better_A_or_B_or_tie", "").strip().upper()
        if better in ("A","B"): wins.setdefault(k[f"output_{better.lower()}_system"], []).append(1); wins.setdefault(k[f"output_{'b' if better == 'A' else 'a'}_system"], []).append(0)
    payload={"systems":{s:{m:{"mean":mean(v),"bootstrap_95_ci":interval(v)} for m,v in metrics.items()} for s,metrics in values.items()},"win_rate":{s:{"mean":mean(v),"bootstrap_95_ci":interval(v)} for s,v in wins.items()}}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(payload,indent=2))
if __name__ == "__main__": main()
