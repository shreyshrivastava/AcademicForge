# Deployment failure investigation

Observed on 21 October 2026. Scope: the two failed deployment environments in the supplied GitHub screenshot, the current main checks, and the standalone demo's runtime failures. No Render services or GitHub failure records were deleted or relabeled.

## Findings

| Evidence | Finding | Action |
|---|---|---|
| GitHub deployment `7227942197`, environment `main - academicforge-frontend`, SHA `1d1c437`, failed at `2026-10-21T13:55:39Z` | This is the external Render frontend, not a failed unit test or the standalone Streamlit deployment. | Retire its external service integration if it is no longer wanted. |
| GitHub deployment `7227942151`, environment `main - academicforge-backend`, same SHA, failed at `2026-10-21T13:55:41Z` | The separate Render backend is also still connected. | Disable auto-deploy or disconnect it in the owning Render account. |
| [Checks run 39611458471](https://github.com/shreyshrivastava/AcademicForge/actions/runs/39611458471), SHA `1d1c437` | The latest checked main revision passed. The older `ab87a7d` run contains three successful jobs and one cancelled macOS 3.14 job. | Do not mistake the old aggregate failure or Render entries for a current application test failure. |
| The configured repository webhook is Streamlit's `https://share.streamlit.io/hook`. No Render repository webhook was returned. | Removing that webhook would affect Streamlit, not retire the Render services. Render may be connected through an external GitHub app/service configuration. | Preserve the Streamlit integration. |
| Standalone dependency and runtime tests | The new demo deliberately excludes neural model packages, bypasses their loaders and uses shared BM25 search. Repeated previews no longer encounter the old one-use quota. | Keep the standalone CI profile and explicit capability labels. |

GitHub's connected tools expose commit status and workflow checks but not the required Render deployment/service management or private logs. The GitHub deployment API was used to identify the screenshot's exact environments. The Render build exception, service configuration and account access were not available, so the precise build failure is **unconfirmed**. No claim is made that a dashboard service was repaired or disconnected.

The service log links returned by GitHub are:

- [Frontend Render deploy](https://dashboard.render.com/web/srv-d96ii4t8nd3s73bim4qg/deploys/dep-dbcc8cq6iqdc73blrmj0)
- [Backend Render deploy](https://dashboard.render.com/web/srv-d96ii4t8nd3s73bim4q0/deploys/dep-dbcc8cuk7voc73a28eb0)

These require access to the owning Render account. Historical red deployment records can remain after services are disconnected; they should not be falsified as successful. New failures will continue while retired services auto-deploy from GitHub.

## Reproduce the checks

With authenticated GitHub CLI access:

```bash
gh api repos/shreyshrivastava/AcademicForge/deployments/7227942197/statuses
gh api repos/shreyshrivastava/AcademicForge/deployments/7227942151/statuses
gh api repos/shreyshrivastava/AcademicForge/actions/runs/39611458471/jobs
gh api repos/shreyshrivastava/AcademicForge/hooks
```

Evidence → finding → remediation: GitHub reports a Render service failure → retired external services are still subscribed to main → disable those services in Render while retaining the standalone Streamlit deployment. Repository fixes address the app's shared code and lightweight runtime; they cannot change the configuration of an inaccessible external service.

## Unified app validation

The standalone profile now shares the current UI and retrieval pipeline and adds quantized FLAN-T5-small on CPU. It retains an extractive-analysis fallback, explicit source warnings, and useful evidence-specific plans. Fresh lightweight environments were checked on Python 3.11 and 3.14, including model loading and real generation. No Torch, sentence-transformers, MLX, FastAPI or OpenAI package is required for the standalone profile.

The real local main search returned HTTP 200 and ten computing papers for `rag as`. The standalone search also returned ten, led by the RAGAS evaluation paper. Both retained arXiv results when Semantic Scholar returned HTTP 429. A separate real-paper AppTest completed search, small-model summary and plan without UI exceptions. [Smoke observations](diagnostics/small-model-smoke.json) record the model revision and local memory measurements; these are not hosted performance guarantees or summary-quality scores.

Final local verification: **128 full-profile tests passed**. The **21 standalone contract tests passed on Python 3.11 and 3.14**, with model packages excluded from unit inference. Real ONNX loading and generation were tested separately. Ruff, compilation and dependency validation passed. The existing upstream Starlette TestClient deprecation warning is non-fatal.
