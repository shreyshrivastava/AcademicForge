# Reliability changes

This document records the work following the [21 October engineering diagnostic](2026-10-21_engineering-AcademicForge-report.md). That report describes commit `73832e7`; its findings are historical, and the current acceptance suite is `tests/test_resilience.py`.

## Changes implemented

| Diagnostic area | Current implementation |
| :--- | :--- |
| Startup import failure | Startup no longer runs a path-dependent preload script. Explicit downloads use `python -m scripts.download_weights`. |
| Cloud opt-out | Python configuration resolves providers. Explicit Fireworks model settings are rejected unless cloud use is enabled and keyed. The launcher no longer overrides model selection. |
| Blocking requests | Search and generation run in workers behind bounded admission. Cancellation retains the generation slot until underlying work finishes. |
| Stream recovery | NDJSON events explicitly end in `done` or `error`. Neither the server nor frontend appends a restarted answer after partial output. |
| Readiness | Lifespan warmup records generation, dense retrieval, and reranker states independently. `/live` is independent of model readiness. |
| Launcher lifecycle | Readiness has a deadline, curl calls have timeouts, the launcher's instance is checked, backend exits are detected, and both child processes are cleaned up with a bounded grace period. |
| Input and errors | Query, paper, summary, mode, category, and body limits apply before expensive work. Public errors contain stable codes and request IDs instead of raw exceptions. |
| Source failure | Individual source statuses and warnings distinguish partial results from total outages. HTTP source calls use bounded attempts and timeouts; expansion does not repeatedly hammer a failed source. |
| OpenJev | Typed response validation rejects malformed shapes and out-of-range scores while preserving existing results. |
| Generation fallback | Both Groq and Fireworks can recover to an authorized local backend before output begins. Responses expose the actual provider/model and fallback use; the UI displays a fallback notice. |
| Dependency/release checks | A universal `uv.lock`, generated hash-pinned pip exports, a Python version policy, and a Linux/macOS CI matrix replace unbounded dependency selection. |
| Resource boundaries | Loopback default, an optional API key required by the launcher for network exposure, per-process rate/body limits, and inference admission protect a small deployment. |
| UI maintenance | Presentation components and CSS have separate files. Pure workflow cache keys include evidence, models, mode, and prompt revision. Summaries are reused and plan text renders incrementally. |
| Managed hosting | No Render.com configuration, hosting URL, or provider-specific runtime path existed in the reviewed checkout. Setup instructions cover local processes and GPU VMs. No external hosting account was modified. |

## Verification

The unit suite uses fake models and providers, isolates developer credentials, and includes startup failure/cleanup, source outage, malformed response, cancellation, capacity, authentication, body/rate-limit, frontend stream, and initial Streamlit page checks. It does not download model weights or call a paid provider.

Local verification on 21 October 2026:

- Existing development environment: **82 tests passed**.
- Isolated environment synchronized from `uv.lock`: **82 tests passed**; all 91 installed packages passed dependency checks.
- Ruff correctness checks, Python compilation, shell syntax, documentation links, and `git diff --check` passed.
- A real import of the pinned MLX stack succeeded and restored `AutoTokenizer.register`; no generation weights were loaded.
- The isolated setup reused matching installed PyTorch/MLX artifacts to avoid repeating large downloads. Other dependencies were installed from the lock. This verifies the locked environment, not a GPU inference workload.
- One upstream Starlette warning remains about future TestClient transport migration; it does not fail the suite.

The CI workflow has been added for future pushes; adding the workflow does not mean a remote CI run has already passed. Run the local checks in [the setup guide](setup.md#development-and-dependency-updates).

## Deployment limits

- GPU installation deliberately requires a compatible preconfigured PyTorch 2.12.x image. CUDA/ROCm driver and model inference validation must be done on that image. Local tests do not establish GPU compatibility.
- The backend key does not authenticate visitors to Streamlit. A shared/public UI needs an authenticated HTTPS gateway and firewall rules.
- Rate limits, caches, and admission counters are per process. Use one inference process per GPU unless capacity has been measured, and apply shared gateway limits when scaling.
- Local inference cannot safely be force-killed inside a Python thread. If a client disconnects, the slot remains occupied until the operation finishes. The launcher can terminate the process after its shutdown grace period; a supervised worker-process/job design is the next step for hard per-job execution deadlines.
- Source HTTP calls have a shared budget for starting attempts plus connect/read inactivity timeouts; they are not a guarantee against every slow-response network behavior. Real network latency and rate limits need integration testing.
- Job state and generated outputs remain session-local. Markdown export is supported; durable saved jobs and resume across restarts are not implemented.
- Readiness reports warmup component state. A model that becomes usable later may need a restart to refresh the startup readiness snapshot.
- The scoped MLX compatibility guard remains necessary for the pinned mlx-lm release and is documented in setup. It restores the public registration method immediately after import.

These changes provide a stronger engineering baseline. A production release still needs actual target-hardware testing, deployment access controls, and operational monitoring appropriate to its users and workload.

## Retrieval quality evaluation

A measured [BEIR SciFact BM25 benchmark](benchmarks/README.md) now supplements the software tests. It covers all 300 test queries over 5,183 abstracts and reports nDCG, recall, MRR, precision, hit rate, timing, and query-bootstrap intervals. Synthetic contract scores are labeled as fixtures; the perfect relevance-score fixture was removed. Neither unit-test pass rate nor a provider confidence value is reported as measured model accuracy.
