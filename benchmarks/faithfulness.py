"""Measure whether generated summary sentences are supported by paper abstracts.

This is a reproducible *screening* check, not a replacement for expert review.
For each generated sentence, it finds the most semantically similar sentence in
the retrieved abstracts using BGE-small embeddings. A sentence is marked
supported when that cosine similarity meets the configured threshold.

Run only after local model weights are available. The script writes its JSON
artifact only when every requested model completes successfully:

    python -m benchmarks.faithfulness --device cpu

If FIREWORKS_API_KEY is available, the script also evaluates the configured
Fireworks DeepSeek model. Set FIREWORKS_MODEL to select a supported endpoint.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import tempfile

from backend.config import get_config
from backend.data_pipeline import retrieve_and_rank_papers
from backend.retrieval.device import select_retrieval_device
from backend.summarizer import summarize_paper


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERIES = ROOT / "benchmarks" / "queries.json"
DEFAULT_OUTPUT = ROOT / "docs" / "benchmarks" / "faithfulness.json"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_THRESHOLD = 0.62
CONTROL_THRESHOLDS = [round(0.50 + index * 0.05, 2) for index in range(7)]


def split_sentences(text: str) -> list[str]:
    """Split prose conservatively so each checked unit remains reviewable."""
    normalized = " ".join((text or "").split())
    if not normalized:
        return []
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", normalized)
        if sentence.strip()
    ]


def dot(left, right) -> float:
    return float(sum(float(a) * float(b) for a, b in zip(left, right)))


def score_sentence_embeddings(summary_sentences, abstract_sentences, summary_embeddings, abstract_embeddings, threshold):
    """Return per-sentence support records from already-normalized embeddings."""
    if len(summary_sentences) != len(summary_embeddings):
        raise ValueError("Summary sentence and embedding counts do not match")
    if len(abstract_sentences) != len(abstract_embeddings):
        raise ValueError("Abstract sentence and embedding counts do not match")

    records = []
    for sentence, embedding in zip(summary_sentences, summary_embeddings):
        if not abstract_sentences:
            records.append(
                {
                    "sentence": sentence,
                    "supported": False,
                    "max_similarity": None,
                    "supporting_abstract_sentence": "",
                }
            )
            continue
        scores = [dot(embedding, candidate) for candidate in abstract_embeddings]
        best_index = max(range(len(scores)), key=scores.__getitem__)
        similarity = scores[best_index]
        records.append(
            {
                "sentence": sentence,
                "supported": similarity >= threshold,
                "max_similarity": similarity,
                "supporting_abstract_sentence": abstract_sentences[best_index],
            }
        )
    return records


class AbstractSupportChecker:
    """BGE embedding-based support checker with an explicit, fixed threshold."""

    def __init__(self, model_name: str, device: str, threshold: float):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required for faithfulness evaluation") from exc

        self.model_name = model_name
        self.device = device
        self.threshold = threshold
        self.model = SentenceTransformer(model_name, device=device)

    def assess(self, abstracts: list[str], summary: str) -> dict:
        summary_sentences = split_sentences(summary)
        abstract_sentences = [sentence for abstract in abstracts for sentence in split_sentences(abstract)]
        if summary_sentences and abstract_sentences:
            summary_embeddings = self.model.encode(summary_sentences, normalize_embeddings=True)
            abstract_embeddings = self.model.encode(abstract_sentences, normalize_embeddings=True)
        else:
            summary_embeddings = []
            abstract_embeddings = []

        sentences = score_sentence_embeddings(
            summary_sentences,
            abstract_sentences,
            summary_embeddings,
            abstract_embeddings,
            self.threshold,
        )
        supported_count = sum(record["supported"] for record in sentences)
        return {
            "sentence_count": len(sentences),
            "supported_sentence_count": supported_count,
            "support_rate": supported_count / len(sentences) if sentences else 0.0,
            "sentences": sentences,
        }

    def assess_thresholds(self, abstracts: list[str], summary: str, thresholds=CONTROL_THRESHOLDS) -> list[dict]:
        """Embed once and retain scores for a fixed, auditable threshold grid."""
        summary_sentences = split_sentences(summary)
        abstract_sentences = [sentence for abstract in abstracts for sentence in split_sentences(abstract)]
        summary_embeddings = self.model.encode(summary_sentences, normalize_embeddings=True) if summary_sentences else []
        abstract_embeddings = self.model.encode(abstract_sentences, normalize_embeddings=True) if abstract_sentences else []
        records = score_sentence_embeddings(summary_sentences, abstract_sentences, summary_embeddings, abstract_embeddings, 1.1)
        return [{"threshold": threshold, "records": [{**record, "supported": record["max_similarity"] is not None and record["max_similarity"] >= threshold} for record in records]} for threshold in thresholds]


def corrupt_summary(summary: str, query: str) -> dict:
    """Create two deterministic, fluent unsupported controls without an LLM."""
    sentences = split_sentences(summary)
    if not sentences:
        return {"unsupported": summary, "reversed": summary, "target_indices": []}
    unsupported = list(sentences)
    unsupported[0] = f"For {query}, the evidence establishes a universal causal effect in every setting."
    reversed_sentences = list(sentences)
    replacements = (("improves", "does not improve"), ("reduce", "increase"), ("increases", "does not increase"), ("more", "less"))
    for index, sentence in enumerate(reversed_sentences):
        lower = sentence.lower()
        for positive, negative in replacements:
            if positive in lower:
                start = lower.index(positive)
                reversed_sentences[index] = sentence[:start] + negative + sentence[start + len(positive):]
                return {"unsupported": " ".join(unsupported), "reversed": " ".join(reversed_sentences), "target_indices": [0, index]}
    reversed_sentences[-1] = "The evidence does not support the conclusion described above."
    return {"unsupported": " ".join(unsupported), "reversed": " ".join(reversed_sentences), "target_indices": [0, len(sentences) - 1]}


def control_summary(records: list[dict], threshold: float) -> dict:
    total = len(records)
    supported = sum(record["max_similarity"] is not None and record["max_similarity"] >= threshold for record in records)
    return {"sentences": total, "supported": supported, "rate": supported / total if total else 0.0}


def run_controls(source: Path, checker: AbstractSupportChecker) -> dict:
    """Score mismatch and deterministic corruption controls for an existing run."""
    artifact = json.loads(source.read_text(encoding="utf-8"))
    entries = artifact["results"]["queries"]
    real, mismatch, unsupported, reversed_records, examples = [], [], [], [], []
    for index, entry in enumerate(entries):
        output = entry["outputs"][0]
        summary = output["summary"]
        own_abstracts = [paper.get("abstract", "") for paper in entry["papers"]]
        other = entries[(index + 1) % len(entries)]
        other_abstracts = [paper.get("abstract", "") for paper in other["papers"]]
        corruption = corrupt_summary(summary, entry["query"])
        real_scores = checker.assess_thresholds(own_abstracts, summary)[0]["records"]
        mismatch_scores = checker.assess_thresholds(other_abstracts, summary)[0]["records"]
        unsupported_scores = checker.assess_thresholds(own_abstracts, corruption["unsupported"])[0]["records"]
        reversed_scores = checker.assess_thresholds(own_abstracts, corruption["reversed"])[0]["records"]
        real.extend(real_scores); mismatch.extend(mismatch_scores)
        unsupported.append(unsupported_scores[0])
        reversed_records.append(reversed_scores[corruption["target_indices"][1]])
        examples.append({"id": entry["id"], "mismatch_query_id": other["id"], "unsupported_summary": corruption["unsupported"], "reversed_summary": corruption["reversed"]})
    rows = []
    for threshold in CONTROL_THRESHOLDS:
        row = {"threshold": threshold, "real_supported": control_summary(real, threshold), "mismatched_supported": control_summary(mismatch, threshold), "unsupported_claim_flagged": {"sentences": len(unsupported), "flagged": sum(record["max_similarity"] is None or record["max_similarity"] < threshold for record in unsupported)}, "reversed_conclusion_flagged": {"sentences": len(reversed_records), "flagged": sum(record["max_similarity"] is None or record["max_similarity"] < threshold for record in reversed_records)}}
        row["unsupported_claim_flagged"]["rate"] = row["unsupported_claim_flagged"]["flagged"] / len(unsupported)
        row["reversed_conclusion_flagged"]["rate"] = row["reversed_conclusion_flagged"]["flagged"] / len(reversed_records)
        row["separation"] = row["real_supported"]["rate"] - row["mismatched_supported"]["rate"] + row["unsupported_claim_flagged"]["rate"]
        rows.append(row)
    best = max(rows, key=lambda row: (row["separation"], row["reversed_conclusion_flagged"]["rate"], -row["threshold"]))
    return {"source_artifact": str(source), "method": "BGE cosine similarity controls", "thresholds": rows, "selected_threshold": best["threshold"], "selection": "Maximized real-vs-mismatch separation plus unsupported-claim flagging; reversed conclusions are reported separately.", "reversed_conclusions_separated": bool(best["reversed_conclusion_flagged"]["rate"] > 0), "examples": examples}


def load_queries(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    queries = payload.get("queries") if isinstance(payload, dict) else payload
    if not isinstance(queries, list) or len(queries) != 20:
        raise ValueError("The faithfulness query set must contain exactly 20 queries")
    if not all(isinstance(item, dict) and item.get("id") and item.get("query") for item in queries):
        raise ValueError("Each faithfulness query must contain an id and query")
    return queries


def requested_models() -> list[dict]:
    config = get_config()
    models = [
        {
            "id": "local_gemma",
            "label": "Local Gemma",
            "provider": config.llm_provider,
            "model": config.model_for_task_and_mode("summary", "fast"),
            "mode": "fast",
        }
    ]
    if os.getenv("FIREWORKS_API_KEY"):
        # This process explicitly opted into the optional cloud comparison.
        os.environ["LOCAL_LLM_USE_FIREWORKS"] = "true"
        cloud_config = get_config()
        models.append(
            {
                "id": "fireworks_deepseek",
                "label": "Fireworks DeepSeek",
                "provider": "fireworks",
                "model": cloud_config.model_for_task_and_mode("summary", "deep"),
                "mode": "deep",
            }
        )
    return models


def paper_record(paper: dict) -> dict:
    return {
        "paper_id": paper.get("paper_id"),
        "title": paper.get("title"),
        "url": paper.get("url") or paper.get("link"),
        "source": paper.get("source"),
        "abstract": paper.get("abstract"),
    }


def evaluate(queries: list[dict], checker: AbstractSupportChecker, paper_limit: int) -> dict:
    results = []
    models = requested_models()
    total = len(queries)
    for index, item in enumerate(queries, 1):
        if index % 5 == 0 or index == 1 or index == total:
            print(f"[faithfulness] Query {index}/{total}: {item['id']} ({item['query'][:40]}...)", flush=True)
        papers = retrieve_and_rank_papers(item["query"])
        papers = papers[:paper_limit]
        if not papers:
            raise RuntimeError(f"No papers were retrieved for faithfulness query {item['id']!r}")
        abstracts = [paper.get("abstract", "") for paper in papers]
        query_result = {
            "id": item["id"],
            "query": item["query"],
            "papers": [paper_record(paper) for paper in papers],
            "outputs": [],
        }
        for model in models:
            # The existing app summarizer is deliberately used unchanged.
            summary = summarize_paper(papers[0], model=model["model"], mode=model["mode"])
            assessment = checker.assess(abstracts, summary)
            query_result["outputs"].append({**model, "summary": summary, "support": assessment})
        results.append(query_result)

    all_outputs = [output for item in results for output in item["outputs"]]
    by_model = {}
    for model in models:
        outputs = [output for output in all_outputs if output["id"] == model["id"]]
        sentence_count = sum(output["support"]["sentence_count"] for output in outputs)
        supported_count = sum(output["support"]["supported_sentence_count"] for output in outputs)
        by_model[model["id"]] = {
            "outputs": len(outputs),
            "sentences": sentence_count,
            "supported_sentences": supported_count,
            "micro_support_rate": supported_count / sentence_count if sentence_count else 0.0,
        }
    return {"models": models, "summary": by_model, "queries": results}


def write_json_atomically(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")
        temporary_path = Path(file.name)
    temporary_path.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--paper-limit", type=int, default=3)
    parser.add_argument("--controls-from", type=Path, help="Run controls against an existing faithfulness artifact")
    parser.add_argument("--controls-output", type=Path, default=ROOT / "docs" / "benchmarks" / "faithfulness_controls.json")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if args.paper_limit < 1:
        raise ValueError("--paper-limit must be at least 1")

    os.environ["ACADEMICFORGE_RETRIEVAL_DEVICE"] = args.device
    device = select_retrieval_device()
    queries = load_queries(args.queries)
    checker = AbstractSupportChecker(args.embedding_model, device, args.threshold)
    if args.controls_from:
        controls = run_controls(args.controls_from, checker)
        controls["generated_at"] = datetime.now(timezone.utc).isoformat()
        controls["environment"] = {"platform": platform.platform(), "python": platform.python_version(), "embedding_device": device}
        write_json_atomically(args.controls_output, controls)
        print(f"Wrote faithfulness control artifact: {args.controls_output}")
        return
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": {
            "type": "embedding_sentence_support",
            "embedding_model": args.embedding_model,
            "threshold": args.threshold,
            "limitations": (
                "Cosine similarity is a heuristic: it can miss paraphrased support and can mark a semantically "
                "similar but contradicted sentence as supported. It does not replace human or NLI-based review."
            ),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "embedding_device": device,
        },
        "results": evaluate(queries, checker, args.paper_limit),
    }
    write_json_atomically(args.output, payload)
    print(f"Wrote faithfulness artifact: {args.output}")


if __name__ == "__main__":
    main()
