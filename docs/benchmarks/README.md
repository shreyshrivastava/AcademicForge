# Retrieval benchmark

## Ablation status

### Pre-registered retrieval decision rule

Before inspecting the frozen SciFact test result, AcademicForge will make fusion
the default only when its paired 1,000-resample query-bootstrap 95% interval for
the nDCG@10 difference versus BGE dense-only excludes zero in the positive
direction. It will make cross-encoder reranking the default only when its
increment over the selected fusion also meets that criterion and the measured
prepared-index latency is justified for the interactive application. Otherwise
the corresponding component remains opt-in. SciFact has no development qrels,
so the official train qrels choose the configuration and the test qrels are used
once for this decision.

AcademicForge has a reproducible ablation runner for the production retrieval classes: `BM25Index`, `DenseIndex`, reciprocal rank fusion, and the BGE cross-encoder reranker. It evaluates BEIR SciFact and NFCorpus, including the BM25 stemming-plus-stopword-removal variant.

The cache-first SciFact evaluation completed locally on Apple Silicon MPS. Its
configuration sweep used the official SciFact train qrels because SciFact ships
no development qrels; the official test qrels were evaluated once after that
choice was frozen. NFCorpus has not been run.

<!-- BEGIN GENERATED ABLATION TABLE -->
| Dataset | Dense default | Best score fusion | Fusion + reranker@20 |
| :--- | :--- | :--- |
| BEIR SciFact | 0.7200 | 0.7306 (not significant) | 0.7278 (not significant) |
| BEIR NFCorpus | 0.3374 | 0.3551 (significant) | Not run by design |
<!-- END GENERATED ABLATION TABLE -->

### Frozen NFCorpus replication

NFCorpus was evaluated once on its official test split with the configuration
frozen on SciFact: weighted RRF uses dense weight 2 and `k=10`; score fusion
uses dense alpha 0.9. There was no NFCorpus tuning and no cross-encoder run.

| System | nDCG@10 (95% CI) | Difference vs dense-only (95% CI) |
| :--- | :--- | :--- |
| Dense-only | 0.3374 (0.3014–0.3752) | — |
| Stemmed BM25 | 0.3053 (0.2737–0.3390) | −0.0322 (−0.0530–−0.0100), significant worse |
| Equal RRF | 0.3422 (0.3082–0.3790) | +0.0048 (−0.0100–0.0212), not significant |
| Frozen weighted RRF | 0.3505 (0.3152–0.3866) | +0.0130 (0.0047–0.0205), significant |
| Frozen score fusion | 0.3551 (0.3192–0.3930) | +0.0177 (0.0108–0.0253), significant |

The metrics convert NFCorpus’s graded qrels to binary relevance: each judgment
above zero is relevant and grade magnitude is not used. This supports frozen
fusion on NFCorpus but does not override the decision rule: SciFact fusion did
not significantly beat dense-only, so dense remains the default. Together these
two tests cover abstracts only, MPS cache-replay latency, and no human evaluation.
See [the full NFCorpus artifact](nfcorpus-frozen-evaluation.json).

Reproduce the cache-first SciFact sweep and frozen test evaluation on MPS:

```bash
python -m benchmarks.fusion_experiments --phase all --progress
```

The command writes [fusion-sweep.json](fusion-sweep.json) and
[retrieval-fusion-evaluation.json](retrieval-fusion-evaluation.json). The latter
contains every query-level metric, 1,000-resample paired bootstrap comparisons,
device metadata, cache provenance, checksums, and source fingerprints.

## What this shows

Dense-only is the default because the score-fusion improvement over dense-only
has a paired nDCG@10 interval of −0.0037 to 0.0268, which includes zero. The
reranker also did not improve its selected fusion (−0.0273 to 0.0211). Hybrid
and reranking modes remain opt-in through `ACADEMICFORGE_RETRIEVAL_MODE`.

The evaluation artifact also records p50/p95 latency for each cached score-array
operation on MPS: dense 0.00050/0.00079 ms, stemmed BM25
0.00046/0.00063 ms, equal RRF 0.00008/0.00017 ms, weighted RRF
0.00008/0.00013 ms, score fusion 0.00008/0.00017 ms, and cached rerank ordering
0.00008/0.00013 ms. These are real cache-replay timings from the script, not
end-to-end model inference or application response latency; they are included
only to compare NumPy fusion operations and cannot be compared with CUDA or ROCm.

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

`benchmarks/faithfulness.py` retrieves papers for 20 fixed questions, runs the same summary function used by the application, and compares each generated sentence with sentences from the selected abstracts using normalized `BAAI/bge-small-en-v1.5` embeddings. This is a screening heuristic, not factuality verification: semantic similarity can accept a related but contradicted claim.

The original local-Gemma run contains 20 outputs and 63 sentence-level screening decisions, but “63/63 supported” is not a factuality claim. Run the deterministic mismatch and corruption controls before interpreting it:

```bash
python -m benchmarks.faithfulness --device cuda --controls-from docs/benchmarks/faithfulness.json
```

The resulting [control artifact](faithfulness_controls.json) reports real-support, mismatched-support, injected-claim flagging, and reversed-conclusion flagging at thresholds 0.50 through 0.80. On the local MPS run, the selected 0.80 threshold marked 65.1% of real sentences supported, 0% of mismatched sentences supported, 100% of injected unsupported claims flagged, and 85% of reversed conclusions flagged. This remains a screening heuristic: the 15% of reversed conclusions not flagged show that semantic similarity alone cannot establish factuality. The blind sheet is built with:

```bash
python -m scripts.build_human_eval
```

It uses Gemma versus Fireworks when a Fireworks output is available with `FIREWORKS_API_KEY`; otherwise it uses Gemma versus Gemma with a deterministic corrupted sentence so raters can test whether they catch the corruption. The separate answer key is ignored by Git.

Run it with:

```bash
python -m benchmarks.faithfulness --device mps
```

When `FIREWORKS_API_KEY` is set, the script also evaluates the explicitly enabled Fireworks DeepSeek model. It writes no artifact if any requested model call fails.

## Sources and attribution

The datasets are from the [BEIR distribution](https://github.com/beir-cellar/beir#available-datasets), including [SciFact by Wadden et al.](https://github.com/allenai/scifact), *Fact or Fiction: Verifying Scientific Claims* (2020). Evaluation conventions follow [BEIR](https://github.com/beir-cellar/beir); the independent evaluator is [pytrec_eval / TREC](https://github.com/terrierteam/pytrec_eval).

SciFact's [license notice](https://github.com/allenai/scifact/blob/master/LICENSE.md) attributes claims and evidence annotations under CC BY 4.0, and abstracts from S2ORC under ODC-By 1.0. The downloaded corpora remain in the local cache. Result artifacts include benchmark query text and judgment IDs with this attribution; they contain no BEIR corpus abstracts.
