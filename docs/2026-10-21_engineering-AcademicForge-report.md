# AcademicForge software engineering diagnostic

Reviewed on 21 October 2026 against commit `73832e7`.

This is the historical baseline. See [reliability changes](reliability.md) for the fixes and remaining deployment limits.

## Assessment

AcademicForge has a reasonable prototype architecture and several useful fallback paths. It is not ready for an unattended, shared production service. The biggest gaps are predictable startup, consistent provider routing, safe streaming recovery, concurrency, and verification of failure behavior.

The statement “there are no fallbacks” is incorrect. The more accurate criticism is: **fallbacks exist, but their coverage, visibility, and correctness are inconsistent.** Some failures are hidden; others bypass the fallback entirely. Passing the current tests does not demonstrate production readiness.

This is a source review plus offline diagnostics of the local checkout, not an audit of a running deployment. No production accounts, real cloud requests, GPU inference, or live paper APIs were exercised. Application code was left unchanged. Findings marked “reproduced” use controlled mocks or a clean subprocess; static findings describe code paths without asserting an observed production incident.

## What is already good

- Frontend and backend are separate, and retrieval has small modules for keyword ranking, embeddings, fusion, reranking, and device selection.
- Pydantic validates basic request structure.
- There are tests for API behavior, ranking, routing, prompts, and caching: all 34 collected tests passed in this environment.
- Cloud Deep generation is optional in the Python configuration layer.
- Secrets are excluded by `.gitignore`, and `.env.example` provides a configuration starting point.
- Caches have entry limits; the search cache also has a TTL and returns copies.
- Several fallback paths preserve useful functionality rather than failing the whole request.

A professional version can retain the existing Streamlit/FastAPI structure. Microservices, Kubernetes, and a database are not prerequisites for fixing these problems.

## Verification performed

| Check | Result | Limits |
| :--- | :--- | :--- |
| Existing suite, with Hugging Face offline mode | 34 passed, 5 warnings | Does not exercise actual cloud availability, GPU failures, or real deployment load. |
| Installed dependency metadata | `pip check`: no broken requirements | This is dependency consistency, not a vulnerability or reproducibility assessment. |
| Python compilation | Passed for backend, frontend, scripts, and tests | Syntax validation only. |
| Shell syntax | `bash -n start.sh` passed | Does not verify lifecycle behavior. |
| Controlled resilience diagnostics | Eight defects reproduced; one working fallback verified | No downloads or paid API calls; models/providers are mocked except imports. |

Environment: Python 3.14.3; FastAPI 0.139.0; Pydantic 2.13.4; Starlette 1.3.1; Transformers 5.13.0; PyTorch 2.12.1; Streamlit 1.59.0. These are observed installed versions, not a recommended dependency set.

Reproduce from the repository root:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 venv/bin/python -m pytest -q tests
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 venv/bin/python docs/diagnostics/resilience_checks.py
venv/bin/python -m pip check
venv/bin/python -m compileall -q backend frontend scripts tests
bash -n start.sh
```

The diagnostic script disables `.env` loading during app import and uses dummy credentials. Its `CONFIRMED` status means a defect was reproduced; it is not an acceptance test. Captured results are in [resilience-results.jsonl](diagnostics/resilience-results.jsonl), with the reproduction code in [resilience_checks.py](diagnostics/resilience_checks.py).

## Existing fallback coverage

| Failure | Current behavior | Assessment |
| :--- | :--- | :--- |
| One paper source fails | Source returns an empty list; the other source can still contribute. | Useful partial recovery, but source failure is not communicated to the caller. |
| Dense model GPU load fails | Attempts CPU; unavailable model or encoding failure uses lexical cosine scoring. | Useful recovery; model-load failure can remain cached as `None` until restart/cache clear. |
| Cross-encoder load/prediction fails | Keeps the previous ranking; GPU load failure first tries CPU. | Useful recovery; API does not clearly state the ranking was degraded. |
| OpenJev request fails | Preserves original papers for caught exceptions. | Works for tested request failures; malformed JSON shapes can escape handling. |
| Remote Deep generation fails before returning a complete answer | Falls back to the configured local model. | Working in the non-streaming diagnostic. Local generation must itself be available. |
| Remote stream fails after emitting text | Appends a new local answer after the partial remote answer. | Recovery corrupts the combined output. |
| Fireworks generation fails | Exception propagates to the API. | No equivalent local runtime fallback. |
| Frontend connection/timeout failure | Retries up to three times. | Helpful before output begins; unsafe after partial stream output and potentially costly for generation POSTs. |
| Local model load or inference fails | Request fails; startup may wait indefinitely. | No defined degraded operating mode or bounded recovery contract. |

## Findings and fixes

P1 means fix before relying on the app as a shared or unattended service. P2 means address during production hardening. P3 means maintainability improvement. Priorities reflect this project and deployment intent, not a claim of a formal certification standard.

### 1. P1: The default startup preload command fails in a clean environment

**Evidence:** [start.sh](https://github.com/shreyshrivastava/AcademicForge/blob/73832e7/start.sh#L37) executes `python scripts/download_weights.py`. The script imports `backend` at [scripts/download_weights.py:6](https://github.com/shreyshrivastava/AcademicForge/blob/73832e7/scripts/download_weights.py#L6), but direct script execution places `scripts/` on the import path, not the repository root.

**Reproduced:** Running that entrypoint with `--help` and no custom `PYTHONPATH` exits with `ModuleNotFoundError: No module named 'backend'`. No weights need to download to reproduce it.

**Impact:** Preloading is enabled by default. With a configured local model, `set -e` exits startup before Uvicorn launches. A developer's shell configuration can hide this defect.

**Fix:** Execute the helper as a module (`python -m scripts.download_weights`) or package the project and install it. Add a clean-environment startup smoke test. Preloading optional ranking models should not prevent an explicitly supported degraded startup.

### 2. P1: Fireworks can be selected despite explicit opt-out

**Evidence:** [start.sh:26](https://github.com/shreyshrivastava/AcademicForge/blob/73832e7/start.sh#L26) assigns Fireworks Deep and research-plan model names whenever a key exists, regardless of `LOCAL_LLM_USE_FIREWORKS`. [backend/config.py:118](https://github.com/shreyshrivastava/AcademicForge/blob/73832e7/backend/config.py#L118) accepts these explicit model settings. [backend/llm.py:160](https://github.com/shreyshrivastava/AcademicForge/blob/73832e7/backend/llm.py#L160) routes any `accounts/fireworks/` model to Fireworks.

**Reproduced:** With a dummy Fireworks key and `LOCAL_LLM_USE_FIREWORKS=false`, the startup selection fragment still chooses the Fireworks model. A mocked generation call reaches the Fireworks function.

**Impact:** A user expecting local processing can send paper context to a cloud provider and incur charges. Python configuration alone behaves differently from startup through the shell script.

**Fix:** Resolve configuration in one place. Require explicit authorization for cloud routing, validate model/provider compatibility, and test the combination of key present plus cloud disabled. Resolve and display the effective provider before serving requests.

### 3. P1: Search and generation block the API event loop

**Evidence:** [backend/app.py:149](https://github.com/shreyshrivastava/AcademicForge/blob/73832e7/backend/app.py#L149), [165](https://github.com/shreyshrivastava/AcademicForge/blob/73832e7/backend/app.py#L165), [184](https://github.com/shreyshrivastava/AcademicForge/blob/73832e7/backend/app.py#L184), and [203](https://github.com/shreyshrivastava/AcademicForge/blob/73832e7/backend/app.py#L203) define asynchronous routes that directly call synchronous network, ranking, and generation functions.

**Reproduced:** A mocked 200 ms search delayed a separate 10 ms timer on the same event loop to approximately 206 ms. This isolates the scheduling defect without hardware or network variation.

**Impact:** One long search or generation can delay other users and health probes on the single Uvicorn worker. Changing a function to `async def` does not make its internals asynchronous.

**Fix:** Move blocking I/O out of the event loop. Use synchronous route functions or explicit thread offloading where appropriate, then add a bounded inference queue/semaphore. Heavy inference needs capacity limits; merely adding threads or workers can exhaust GPU memory. Test concurrent generation and health requests.

This follows FastAPI's [guidance on blocking libraries and route execution](https://fastapi.tiangolo.com/async/).

### 4. P1: Stream recovery can combine two different answers

**Evidence:** [backend/llm.py:211](https://github.com/shreyshrivastava/AcademicForge/blob/73832e7/backend/llm.py#L211) starts remote streaming, then yields a new local answer on any exception. [frontend/api_client.py:59](https://github.com/shreyshrivastava/AcademicForge/blob/73832e7/frontend/api_client.py#L59) retries a request even after yielding text to its caller.

**Reproduced:** Server output became `REMOTE PARTIAL. LOCAL FULL PLAN.` after a simulated upstream failure. Client output became `PARTIAL. FULL PLAN.` after a simulated read timeout and retry.

**Impact:** Users can receive duplicated sections, conflicting recommendations, and repeated inference charges. The caller cannot identify where the restart happened.

**Fix:** Retry or switch providers automatically only before emitting output. Once output starts, send a typed stream error with explicit restart semantics. Use structured events such as `start`, `delta`, `error`, and `done`, with a request ID. The UI must replace a failed attempt when restarting rather than append a new answer to it.

### 5. P1: Startup can wait forever, and readiness is incomplete

**Evidence:** [start.sh:47](https://github.com/shreyshrivastava/AcademicForge/blob/73832e7/start.sh#L47) loops without a deadline, does not check whether `BACKEND_PID` died, and calls curl without a maximum duration. There is no cleanup trap for the background backend. [backend/app.py:69](https://github.com/shreyshrivastava/AcademicForge/blob/73832e7/backend/app.py#L69) catches startup warmup failures, while [91](https://github.com/shreyshrivastava/AcademicForge/blob/73832e7/backend/app.py#L91) infers readiness only from a generation model cache entry.

**Reproduced:** After simulating successful generation-model loading and failed retrieval warmup, `/health` still returned `status: ok` and `models_ready: true`. The infinite wait and missing process cleanup are static findings.

**Impact:** A failed backend can leave startup hanging. The service can report healthy without reporting failed retrieval components, and the backend can outlive its frontend launcher.

**Fix:** Add a finite startup deadline, curl timeouts, child-process checks, and shutdown cleanup. Separate liveness from readiness. Track per-component states and errors, including a deliberate `degraded` state when lexical retrieval remains usable. Mandatory component failures should produce a readiness failure, not an endless “warming up.” Use FastAPI's [lifespan mechanism](https://fastapi.tiangolo.com/advanced/events/) for initialization and cleanup.

### 6. P2: Invalid and unbounded requests reach expensive operations

**Evidence:** [backend/app.py:25](https://github.com/shreyshrivastava/AcademicForge/blob/73832e7/backend/app.py#L25) permits blank/unbounded queries, arbitrary categories and generation modes, and unbounded paper/summary lists. `PapersRequest` accepts no evidence. Routes have no declared response models.

**Reproduced:** A whitespace-only query and an empty research-plan request with an invalid mode both reached mocked business logic and returned HTTP 200.

**Impact:** Invalid work consumes compute. Large input can exceed context limits or memory. Clients cannot rely on a strongly defined success/error schema.

**Fix:** Add trimmed nonempty query validation, input length and paper-count limits, allowed-value enums, and output schemas. Validate summary-to-paper alignment and token budgets. Return clear validation errors before calling retrieval or inference.

### 7. P2: Upstream outages are reported as insufficient evidence

**Evidence:** [backend/data_pipeline.py:119](https://github.com/shreyshrivastava/AcademicForge/blob/73832e7/backend/data_pipeline.py#L119) and [177](https://github.com/shreyshrivastava/AcademicForge/blob/73832e7/backend/data_pipeline.py#L177) collapse source errors into empty results. Sources run sequentially at [229](https://github.com/shreyshrivastava/AcademicForge/blob/73832e7/backend/data_pipeline.py#L229), and low result counts trigger another fetch at [497](https://github.com/shreyshrivastava/AcademicForge/blob/73832e7/backend/data_pipeline.py#L497). [backend/app.py:156](https://github.com/shreyshrivastava/AcademicForge/blob/73832e7/backend/app.py#L156) returns a normal empty search response.

**Impact:** If both providers fail, the user sees a lack-of-evidence message rather than a temporary service failure. The pipeline can repeat failing calls during expansion. The cache only serves fresh matching searches; there is no intentional stale-result recovery policy.

**Fix:** Return per-source status and distinguish no matches, partial results, and total upstream failure. Give each source a timeout within a total search deadline. Apply bounded backoff for transient failures and rate limits. Consider stale cached results with a visible timestamp when live search fails. Do not silently present stale data as fresh.

Static finding; real provider outages were not induced.

### 8. P2: OpenJev fallback does not cover malformed response shapes

**Evidence:** [backend/jev.py:183](https://github.com/shreyshrivastava/AcademicForge/blob/73832e7/backend/jev.py#L183) assumes response JSON has a `.get` method; the exception handler catches request, value, and type errors but not attribute errors.

**Reproduced:** A successful response containing valid JSON `[]` raised an uncaught `AttributeError` instead of returning the original ranking.

**Fix:** Validate the response against a typed schema before using it. Treat invalid external responses as a dependency failure, preserve the existing ranking, and report degradation. Test lists, nulls, missing fields, and invalid scores.

### 9. P2: Timeout, retry, and fallback policies are inconsistent

**Evidence:** [frontend/api_client.py:75](https://github.com/shreyshrivastava/AcademicForge/blob/73832e7/frontend/api_client.py#L75) retries connection/timeouts for all methods but not HTTP 429/503. Both cloud SDK clients at [backend/llm.py:405](https://github.com/shreyshrivastava/AcademicForge/blob/73832e7/backend/llm.py#L405) and [418](https://github.com/shreyshrivastava/AcademicForge/blob/73832e7/backend/llm.py#L418) rely on SDK timeout/retry defaults rather than an application deadline. Fireworks generation has no local error fallback at [165](https://github.com/shreyshrivastava/AcademicForge/blob/73832e7/backend/llm.py#L165).

**Impact:** A timed-out frontend POST can start another generation while the original is still running. Independent retries at different layers can multiply work. A cloud failure may spend much of the user-facing timeout before local recovery begins. Configuring an alternative provider is not the same as implementing runtime failover.

**Fix:** Define a total request budget and one retry owner per dependency. Retry only classified transient failures, respect `Retry-After`, and use bounded backoff with jitter. Add idempotency/job identifiers for expensive generation. Expose the actual provider and fallback reason in result metadata. Keep cloud fallback opt-in; reject or return a clear unavailable response when no permitted fallback exists.

### 10. P2: Fresh installations and releases are not reproducible

**Evidence:** [requirements-local.txt](https://github.com/shreyshrivastava/AcademicForge/blob/73832e7/requirements-local.txt) uses lower bounds and unpinned major packages. There is no tracked lockfile, declared supported Python version, or CI workflow. GPU requirement files add an extra index without a verified wheel/version constraint. [backend/llm.py:12](https://github.com/shreyshrivastava/AcademicForge/blob/73832e7/backend/llm.py#L12) patches a private Transformers API globally at import and suppresses patch failures.

**Impact:** Two installations may use different dependency behavior. The compatibility patch is coupled to private internals. A green local environment does not establish that a new checkout or a different GPU image works.

**Fix:** Declare supported Python and dependency versions, maintain reproducible per-platform constraints/locks, and test clean installs in CI. Keep MLX dependencies specific to supported Apple hardware. Prefer compatible released dependencies over unconditional private API patches. Record model revisions and tested GPU runtime versions.

No claim is made that the current installed packages are vulnerable; a vulnerability inventory was outside this review.

### 11. P2: Tests miss failure paths and documented commands omit tests

**Evidence:** Existing API tests create `TestClient(app)` without a lifespan context, so they do not exercise startup warmup. The reranker test starts with the expected paper already first and can pass when the cross-encoder fails and returns unchanged input. `tests/test_search_cache.py` and `tests/test_jev.py` have no executable test runner; invoking them as scripts does not collect their tests. Some other scripts' `__main__` lists omit test functions.

**Impact:** The README's individual `python tests/...` commands provide weaker coverage than `pytest` collection. All 34 tests passing coexists with the eight reproduced defects above.

**Fix:** Make `pytest` the single test entrypoint and run it in CI. Add failure-injection cases for source timeouts/429s, startup failures, provider opt-out, partial streams, malformed external responses, invalid input, and concurrent requests. Separate unit tests with fake models from marked hardware/network integration tests. Test reranking with deliberately wrong initial ordering and assert which backend actually ran.

### 12. P2: A public deployment lacks basic resource and error boundaries

**Evidence:** [start.sh:42](https://github.com/shreyshrivastava/AcademicForge/blob/73832e7/start.sh#L42) binds the backend to all interfaces. The app defines no authentication, rate limiting, per-user quotas, or inference admission control. Exception handlers such as [backend/app.py:160](https://github.com/shreyshrivastava/AcademicForge/blob/73832e7/backend/app.py#L160) return raw exception strings. There are ordinary logs but no request IDs, latency/error metrics, or fallback counters.

**Impact:** On a publicly reachable deployment without external gateway protections, anyone who reaches the API can consume GPU or cloud capacity. Internal error details can be exposed, and operators cannot reliably track degraded output or correlate failures.

**Fix:** Use loopback defaults for local demos. For shared hosting, add authentication at the app or gateway, request size/rate limits, bounded queueing, and a compute/cost budget. Return stable error codes with a correlation ID; keep detailed tracebacks in logs. Track request latency, dependency failures, fallback use, queue depth, and generation completion. Existing deployment-level protections were not inspected, so this finding is conditional on exposure.

### 13. P3: Presentation, workflow, and state are too tightly coupled

**Evidence:** [frontend/streamlit_app.py](https://github.com/shreyshrivastava/AcademicForge/blob/73832e7/frontend/streamlit_app.py) is approximately 1,385 lines and combines CSS, rendering, network workflow, state, and export. Research plan generation summarizes all selected papers sequentially again at [1315](https://github.com/shreyshrivastava/AcademicForge/blob/73832e7/frontend/streamlit_app.py#L1315); streamed chunks are accumulated but only assigned to the displayed completed result after the loop. State is session-local, with no durable job or result store.

**Impact:** UI changes are harder to test independently. Failed multi-paper generation can waste completed work. A session loss or backend restart has no defined resume behavior. The API's streaming capability does not currently translate into incremental plan rendering in this UI path.

**Fix:** Extract view components and a workflow service incrementally. Reuse completed summaries keyed by content, model, and prompt version. Render deltas as they arrive. Add durable jobs/results only if recovery across sessions is a product requirement; in-memory state is reasonable for an explicitly ephemeral local tool.

## Recommended repair order

1. **Make the normal launch reliable:** repair the preload entrypoint, centralize model/provider configuration, honor cloud opt-out, add startup deadlines and cleanup, and report component readiness honestly.
2. **Make failures safe:** define typed errors and degraded results; repair stream restart behavior; add bounded source/provider policies and request validation.
3. **Protect shared compute:** offload blocking work, limit simultaneous inference, and add access and resource controls for hosted use.
4. **Make releases verifiable:** choose a supported Python/dependency set, add CI, collect tests with pytest, and add failure/concurrency/startup tests.
5. **Improve maintainability and user recovery:** split the UI, reuse completed work, show actual fallback/provider information, and decide whether durable saved jobs are required.

## Acceptance criteria for a professional baseline

- A clean supported checkout starts without shell-specific `PYTHONPATH` changes; startup failure exits with a clear error and no orphan process.
- Cloud-disabled configuration causes zero cloud generation calls, even when cloud credentials are present.
- Search/generation do not stall liveness checks, and overload produces a bounded, explicit response.
- Losing one paper source yields clearly marked partial results; losing both yields an actionable dependency error or clearly marked cached data.
- Ranking failures retain useful results and expose the active degraded mode.
- Stream failure never appends a new full answer to partial output without an explicit reset.
- Invalid/oversized input is rejected before expensive work.
- Every supported provider has tests for timeout, malformed response, and permitted fallback behavior.
- CI verifies clean installation, the full unit suite, formatting/static checks, and a startup smoke test; hardware tests are separate and explicit.
- A shared deployment has an access policy, capacity limits, safe errors, request tracing, and documented shutdown/recovery behavior.

These are observable release conditions. Their implementation can remain a small, well-structured application.
