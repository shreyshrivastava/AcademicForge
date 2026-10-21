# Demo audit and alignment plan

Implementation update: the agreed target is now a standalone, key-free Streamlit profile sharing main's UI and retrieval code. A hosted model backend is outside that target. See [standalone deployment](streamlit_cloud.md) for the implemented behavior and [deployment findings](2026-10-21_deployment-check.md) for the external Render failures. The audit below describes the historical demo commit, before unification.

Audit date: 21 October 2026. Scope: compare GitHub `demo` at `e548bb33f39496e579728cea459779d6be6ab70c` with the running local `main` at `1d1c437d5f17c3df808828805a29caa7149e5c49`. This is an engineering assessment and implementation plan; no deployment or application changes are part of this audit.

The demo is a separate implementation that has drifted from the local product. It is missing eight commits from main and has four commits unique to the demo. Its missing dense retrieval and reranker packages are intentional, but their fallback behavior is presented as a module failure in the logs. Other differences are real defects: small live result sets are replaced with fictional records, the new search field is absent, and the documented LLM switch does nothing. The recommended target is one shared interface and search pipeline, with explicit runtime capabilities and a small hosted adapter.

## What was inspected

- GitHub branch history, source code, dependency manifests, entry points, tests, and deployment instructions.
- An isolated archive of the demo branch, leaving the running main checkout intact.
- The public URL documented by the demo: <https://academicforge-demo.streamlit.app/>. An unauthenticated HTTP request returned 303 and redirected to a login route; the web reader could not open it. This does not establish that the deployed application is broken or identify which revision is deployed. Hosted build/runtime logs and deployment settings remain unverified.
- The configured Hugging Face repository was also queried through the connected tool, which returned not found or authentication required. The Streamlit demo documented on the actual `demo` branch is the primary target of this report.

## Findings and evidence

### Runtime checks

The isolated demo started without an unhandled Streamlit exception. Its existing retrieval and limiter tests passed: **12 passed**. A controlled Streamlit AppTest search and first plan generation also completed. This means the branch is not universally broken; the current tests do not cover its most important evidence and parity problems.

The demo's declared lightweight requirements were installed into a separate Python 3.14.3 environment. They resolved to Streamlit 1.66.0, arxiv 4.0.1, OpenAI 3.35.1, requests 2.34.2 and NumPy 2.5.4. This tests a fresh installation permitted by the manifest, not the deployment's unverified Python 3.11 environment. Exact package versions and results are in [demo-audit-results.json](diagnostics/demo-audit-results.json).

Controlled source fixtures reproduced these behaviors without network or paid-generation calls:

- Both sources returning no papers produced four `synthetic_demo` papers, an empty response message, and no UI search error. The general hosted-demo banner mentions sample fallback, but there is no per-search warning.
- One relevant live-paper fixture was discarded and replaced entirely with synthetic records.
- The missing `sentence_transformers` dependency logged a reranking failure three times across three searches, while each search continued with fallback ranking.
- A first plan completed. A second attempt was blocked and displayed as “The backend returned an error” even though no backend process is involved in this profile.
- Enabling `ACADEMICFORGE_CLOUD_ENABLE_LLM` retained the `cloud_demo` provider and template summary.
- An unmatched direct arXiv ID returned empty papers and an empty message.
- The UI had no Search field selector, and the runtime search function accepted no domain argument.

The source fixture labeled as a live match is a test record, not a claim about a paper retrieved from the public service. AppTest also emits a bare-mode ScriptRunContext warning from the test harness; that is not an application crash.

All file references below describe the audited `demo` commit unless marked main. Reproduce static observations with `git show origin/demo:<path>` after fetching the branch.

| ID | Priority | Evidence | Finding and effect | Required change |
|---|---|---|---|---|
| F1 | High | E1: `git rev-list --left-right --count main...origin/demo` returns `8 4`; direct tree comparison | The demo maintains an older UI and retrieval implementation. It lacks the field selector, RAG/RAGAS expansion, field-aware caches, current styles/components, and recent resilience changes. | Share the main UI and search services; keep deployment-specific behavior in an adapter. |
| F2 | High | E2: `frontend/cloud_demo_runtime.py:74–78` | Fewer than three relevant live results causes replacement of the entire result set with synthetic records, even when live sources worked. Fictional papers can enter plans and exports. | Preserve valid live results. On source failure, use genuine cached evidence with its age shown or show an actionable unavailable state. Offer sample data only as an explicit, prominently labeled demonstration mode. |
| F3 | Medium | E3: lightweight `requirements.txt`; `backend/retrieval/dense.py`; `backend/retrieval/reranker.py` | `sentence_transformers` and Torch are intentionally absent. Dense retrieval uses lexical similarity; reranking catches the missing import, logs an error and preserves order. This is expected degradation, not proof that all search fails. | Declare capabilities at startup, skip unavailable optional stages, show BM25/lexical/no-reranker accurately, and log expected limitations once. Reserve errors for unexpected failures. |
| F4 | High for parity | E4: `frontend/cloud_demo_runtime.py:112–234`; `docs/streamlit_cloud.md:50–63`; repository search for `CLOUD_ENABLE_LLM` | Summary, guidance, and plans are templates. Fast/deep modes do not select different models. The LLM enable flag and token setting appear in documentation, not the demo implementation. | Label previews honestly. For model-backed parity, connect the shared client to a hosted instance of the current backend with authentication and explicit generation budgets. Remove unsupported settings until implemented. |
| F5 | Medium | E5: `frontend/api_client.py:166–178`; `frontend/cloud_usage_limiter.py` | Missing forwarded headers collapses users to `local-dev-client`. The lifetime one-request quota has no reset window and uses ephemeral local SQLite. Its select/update sequence is not an atomic quota reservation. These are reliability risks; exploitation or concurrent overspend was not tested. | Use a trusted identity policy, an atomic persistent quota store for paid generation, a clear reset window, and visible quota status. A lightweight preview should not imply durable per-IP enforcement. |
| F6 | Medium | E6: `backend/data_pipeline.py:112–115`; `frontend/api_client.py:89–124` | Source access uses the older arXiv client defaults and lacks main's shared deadline/retry/source diagnostic layer. The HTTP client can replay POSTs and streams after connection errors. | Carry over bounded source calls, safe errors, cache handling and single-attempt generation. Preserve completion/error stream semantics. |
| F7 | Medium | E7: `requirements.txt:3–7`; `runtime.txt`; absence of `.github` workflow on demo | Cloud dependencies have only lower bounds and the demo requests Python 3.11. Main requires Python 3.12–3.14. A rebuild or simple merge can introduce incompatible dependencies or imports. | Choose a supported shared Python version, create a locked lightweight cloud dependency profile, and test it separately from the model environment. Do not install the entire GPU stack into Streamlit Cloud. |
| F8 | Medium | E8: main `frontend/api_client.py` and `frontend/streamlit_app.py`; demo equivalents | A direct UI copy is insufficient: main expects field-aware search responses, generation metadata and `last_generation`, while the old demo client lacks them. Current main has also removed the demo entry point and routing. | Define a shared client interface and test both implementations against the same search, summary, guidance and plan contracts. |

Static evidence establishes the source behavior at the pinned commits. Hosted occurrences, deployment configuration and resource exhaustion remain unverified. Optional retrieval fallback must not be described as measured dense-model performance. Main's published SciFact results describe its BM25 implementation, not automatically this older demo or the full model pipeline.

## Plan

### 1. Establish a reproducible baseline

Record the deployed branch, commit, Python version and entry point from Streamlit deployment settings. Capture the actual failing-module logs, preserving exception types and removing credentials. Check startup, search, summary, guidance, plan generation and Markdown download against the pinned demo snapshot. Classify each issue as an expected capability limit, source outage, application error or deployment error.

Acceptance: the deployed revision is visible in diagnostics, and each reported module failure has a reproducible cause or is explicitly recorded as unconfirmed.

### 2. Share the current application

Integrate the hosted entry point and adapter into main's architecture. Use one `frontend/streamlit_app.py`, component/style set, query-context module, search pipeline and export behavior. Add a small runtime interface for local/backend access and hosted preview access. It must return the same response shape, carry the domain and original question through plans and exports, and report generation provenance and completion state.

Keep `demo` as a deployment branch generated or synchronized from the shared source, rather than a separately maintained product. Preserve its hosting files during integration; do not blindly replace its API client with either branch's version.

Acceptance: both profiles have the current layout, Search field presets/custom field, Research Lens, selected evidence, stale-result notices, summary/guidance panels and Markdown export. RAGAS in the computing field uses the same interpretation and filtering in both profiles.

### 3. Make hosted limitations explicit and fix evidence handling

Declare the runtime's retrieval and generation capabilities. Carry over main's source deadlines, diagnostics, field-aware cache handling and safe failure states. Keep successful live evidence during partial outages. Replace automatic fictional-paper fallback with a labeled cache or unavailable state; provide an explicit sample-data option only if needed for demonstrations.

Use the lightweight lexical path intentionally on Streamlit Cloud. Remove artificial streaming delays and misleading fast/deep choices when the runtime only produces templates. Distinguish an evidence preview from a model-generated result in the UI and exported plan.

Acceptance: no recurring missing-package error for an intentionally disabled model; no automatic synthetic citations; source outages and ranking capabilities are visible; a failed operation never appears as a completed generation.

### 4. Provide model-backed parity through the current backend

For the closest possible behavior, host the current FastAPI/backend model stack on suitable compute and connect the Streamlit frontend to it. This shares the actual search, ranking, generation and fallback implementations. Configure credentials in the deployment's secret store, enforce authentication, retain single-attempt expensive requests and use an atomic persistent generation budget. Choose model/hardware capacity and hosting spend before provisioning. No paid infrastructure or generation is required for the audit.

Acceptance: the hosted frontend completes the same search → select → summary/guidance → plan → export workflow against the shared backend, reports the actual model/provider and handles interrupted generation safely. An intentionally lightweight preview remains clearly labeled when this backend is unavailable.

### 5. Test both profiles and release with rollback

Add a CI matrix for the lightweight hosted profile and full backend profile. Test startup without model packages, field-filter behavior, partial/all-source outages, empty/malformed responses, generation failures, stream completion, quota resets/concurrency and export provenance. Compare screenshots of the same states at the same viewport. Keep algorithm tests deterministic; use a small scheduled or release-time source smoke test for real network integration.

Run the same pinned, labeled retrieval benchmark for each ranking configuration being claimed, recording its commit and dependencies. Publish nDCG@10, Recall@10 and MRR@10 with query count and corpus size; separately measure latency, source failure rate and workflow completion. Do not imply a BM25 result measures generation quality or dense/reranker quality.

Deploy a reviewed candidate, verify its visible revision and a real end-to-end flow, and retain the previous deployment revision for rollback. Keep the existing locally running app available during the work.

Acceptance: both dependency profiles pass their meaningful checks, supported features match, deliberate runtime differences are documented, and the deployed revision and completion state can be verified.

## Evidence reproduction

```bash
git fetch origin main demo
git rev-list --left-right --count origin/main...origin/demo
git diff --stat origin/demo origin/main
git show origin/demo:frontend/cloud_demo_runtime.py
git show origin/demo:frontend/api_client.py
git show origin/demo:requirements.txt
git show origin/demo:runtime.txt
git grep -n 'CLOUD_ENABLE_LLM\|CLOUD_MAX_TOKENS' origin/demo -- frontend docs
```

These commands are read-only apart from fetching remote references. Branch names can move; use the full audited commit IDs above when reproducing the original comparison.

For the controlled runtime checks, use a fresh archive and a separate environment; the quota probe consumes its first request in that temporary archive. `uv` must be available. The script path below is relative to this repository before changing directories.

```bash
probe_path="$PWD/docs/diagnostics/demo_audit_probe.py"
demo_audit_dir=$(mktemp -d /tmp/academicforge-demo-audit.XXXXXX)
git archive e548bb33f39496e579728cea459779d6be6ab70c | tar -xf - -C "$demo_audit_dir"
uv venv "$demo_audit_dir/audit-env" --python 3.14
uv pip install --python "$demo_audit_dir/audit-env/bin/python" -r "$demo_audit_dir/requirements.txt" pytest
cd "$demo_audit_dir"
"$demo_audit_dir/audit-env/bin/python" "$probe_path"
"$demo_audit_dir/audit-env/bin/python" -m pytest tests/test_cloud_usage_limiter.py tests/test_retrieval.py -q
```

This reproduces the controlled probes on a fresh dependency resolution. The original resolved versions are recorded in the JSON artifact; unbounded requirements can resolve differently in the future. The probe writes its result to `/tmp/academicforge-demo-audit-results.json`.

## Evidence → finding → implementation path

P1, source outage: public source call → no or few relevant candidates (E2, E6) → sample replacement (F2) → ranked fictional evidence → template plan (E4, F4). Replace this path with shared source diagnostics, preserved/cached real evidence, and explicit unavailable/preview states.

P2, optional model: hybrid search → missing `sentence_transformers` (E3) → lexical fallback and reranker exception log (F3) → usable but different ranking. Replace this path with capability selection before model calls and accurate runtime metadata.

P3, feature release: main gains field filtering and safer completion semantics (E1, E8) → independent demo code remains unchanged (F1, F8). Replace this path with shared source, adapter contract tests, and deployment revision checks.
