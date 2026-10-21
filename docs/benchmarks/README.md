# Retrieval benchmark

## Ablation status

AcademicForge has a reproducible ablation runner for the production retrieval classes: `BM25Index`, `DenseIndex`, reciprocal rank fusion, and the BGE cross-encoder reranker. It evaluates BEIR SciFact and NFCorpus, including the BM25 stemming-plus-stopword-removal variant.

No full ablation artifact is published yet. The runner was started on this Apple Silicon machine, but BGE dense retrieval and cross-encoder reranking did not complete within the practical evaluation window. The runner therefore wrote no partial or estimated results. It fails rather than silently treating its lexical dense fallback as a BGE result.

<!-- BEGIN GENERATED ABLATION TABLE -->
| Dataset | BM25 | Stemmed BM25 | BGE dense | RRF | RRF + cross-encoder |
| :--- | :--- | :--- | :--- | :--- | :--- |
| BEIR SciFact | Prior baseline below | Not yet measured | Not yet measured | Not yet measured | Not yet measured |
| BEIR NFCorpus | Not yet measured | Not yet measured | Not yet measured | Not yet measured | Not yet measured |
<!-- END GENERATED ABLATION TABLE -->

Run the complete ablation on an available CUDA, ROCm, MPS, or CPU device:

```bash
python -m benchmarks.run_scifact --download --device cuda
```

The command writes `retrieval-ablation.json` only after every requested system completes. Each dataset result contains nDCG@10, Recall@10, MRR@10, 95% query-bootstrap intervals from 1,000 resamples with seed 42, paired bootstrap comparisons with the no-stemming BM25 baseline, prepared-index p50/p95 latency, environment data, and checksum-verified dataset metadata.

## What this shows

The currently published result validates only the lexical BM25 component. It does not support claims that dense retrieval, RRF, or the cross-encoder improves relevance for this application. The new runner is designed to answer that question without changing application retrieval behavior; publish its table only after a complete result artifact exists.

## Published BM25 baseline

Measured on 21 October 2026 using AcademicForge's **BM25 lexical component** on the complete **BEIR SciFact test split**: **300 queries**, **5,183 abstracts**, and **339 positive query-document judgments**. These are measured retrieval results, separate from synthetic unit-test scores.

| Metric | Measured score | Query-bootstrap 95% interval | Meaning |
| :--- | ---: | :--- | :--- |
| nDCG@10 | **0.6427** | 0.5959-0.6894 | Rewards placing judged relevant papers near the top. |
| Recall@10 | **0.7645** | 0.7201-0.8112 | Fraction of each query's judged relevant papers found in its top ten. |
| MRR@10 | **0.6078** | 0.5573-0.6552 | Rewards finding the first judged relevant paper early. |
| Precision@10 | 0.0843 | 0.0783-0.0903 | Fraction of ten retrieved positions containing a judged relevant paper. |
| Hit rate@10 | 0.7833 | 0.7400-0.8267 | Fraction of queries with at least one judged relevant paper in the top ten. |

Recall@10 of 0.7645 means **76.45% mean recall**, not model accuracy. Precision is much lower because SciFact supplies only about one judged relevant abstract per query, while the denominator is ten; unjudged documents count as nonrelevant under this evaluation protocol.

The measured prepared-index search latency was **5.69 ms p50** and **9.97 ms p95** on this Apple Silicon host, with **0.167 seconds** to construct the index. These timings exclude dataset loading, network calls, model loading, generation, and UI work. They are not application response times or hardware-independent performance guarantees.

## What was evaluated

- Full-corpus title-and-abstract BM25 retrieval using the application's tokenizer and scoring implementation.
- Existing parameters: `k1=1.5`, `b=0.75`; no stemming or stopword removal.
- Every query with positive judgments in the official test split. No sampling, cherry-picking, or tuning on test judgments.
- Binary relevance: a positive official judgment counts as relevant. Other retrieved documents count as nonrelevant.
- A prepared index reused across queries; a single-call application search uses the same implementation.
- Query bootstrap intervals from 1,000 resamples with seed 42. These describe variation over the benchmark queries, not generalization to every subject area.

No local LLM, cloud provider, generated relevance labels, mocked score, or fallback was used to calculate these results. This run does **not** evaluate BGE embeddings, RRF, the cross-encoder, OpenJev, live-source recall, summaries, research-plan quality, or factual correctness. The result must not be presented as the accuracy of the full application or compared as an identical implementation to BEIR's Elasticsearch BM25 baseline.

## Reproduce

From the repository root, using the project environment:

```bash
python -m benchmarks.run_scifact --download --device cpu
```

The first run downloads the official archives into the ignored `.academicforge_cache/benchmarks/` directory. Existing data is checked against the recorded SHA-256 before every evaluation.

[scifact-bm25.json](scifact-bm25.json) contains aggregate metrics, intervals, every query's ranked document IDs and scores, dataset checksum, source-code fingerprints, configuration, environment, and timing. The Git HEAD identifies the base commit; source fingerprints identify the exact implementation used even when the working tree is uncommitted. Re-running refreshes that file and its timing measurements.

For an independent check of the existing SciFact evaluation arithmetic:

```bash
uv run --no-project --python 3.14 --with pytrec-eval-terrier==0.5.11 python -m benchmarks.verify_scifact
```

This reads the original dataset judgments independently and compares every query's nDCG, recall, precision, and reciprocal rank with the TREC reference evaluator. All four metrics matched for all 300 queries. The [reference-check artifact](scifact-reference-check.json) binds this check to the result file's SHA-256. A zero arithmetic difference means agreement between evaluators; it does not mean perfect retrieval.

Unit tests separately check metric arithmetic, duplicate rejection, empty retrievals, macro-averaging, and short-result denominators. The external datasets are not downloaded during CI unit tests. A new result must come from a benchmark run, not from the number of unit tests passing.

## Generated-output faithfulness screen

`benchmarks/faithfulness.py` retrieves papers for 20 fixed questions, runs the same summary function used by the application, and checks each generated sentence against all sentences in the selected abstracts using normalized `BAAI/bge-small-en-v1.5` embeddings. A sentence is flagged as supported when its highest cosine similarity is at least 0.62.

The [faithfulness artifact](faithfulness.json) contains a real local-Gemma run: 20 outputs, 63 generated sentences, and 63 heuristic support decisions. It does not establish factual correctness. Embedding similarity can miss valid paraphrases, and it can accept a semantically related but contradicted sentence. The [blind human-evaluation template](human_eval_template.csv) contains the same 20 query IDs without model labels or ratings.

Run it with:

```bash
python -m benchmarks.faithfulness --device mps
```

When `FIREWORKS_API_KEY` is set, the script also evaluates the explicitly enabled Fireworks DeepSeek model. It writes no artifact if any requested model call fails.

## Sources and attribution

The datasets are from the [BEIR distribution](https://github.com/beir-cellar/beir#available-datasets), including [SciFact by Wadden et al.](https://github.com/allenai/scifact), *Fact or Fiction: Verifying Scientific Claims* (2020). Evaluation conventions follow [BEIR](https://github.com/beir-cellar/beir); the independent evaluator is [pytrec_eval / TREC](https://github.com/terrierteam/pytrec_eval).

SciFact's [license notice](https://github.com/allenai/scifact/blob/master/LICENSE.md) attributes claims and evidence annotations under CC BY 4.0, and abstracts from S2ORC under ODC-By 1.0. The downloaded corpora remain in the local cache. Result artifacts include benchmark query text and judgment IDs with this attribution; they contain no BEIR corpus abstracts.
