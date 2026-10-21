# Standalone Streamlit deployment

The standalone profile runs the current AcademicForge interface entirely inside Streamlit. It needs no API keys, GPU, or separate FastAPI process. A small public model downloads once and is cached; search stays usable while it loads. Both profiles share search fields, source adapters, evidence selection, styles, and export behavior.

| Capability | Local profile | Standalone profile |
|---|---|---|
| Search fields and Research Lens | Shared | Shared |
| Academic sources | arXiv and Semantic Scholar | Same sources |
| Ranking | Hybrid retrieval and optional neural reranking | BM25 |
| Summary, guidance, plan | Configured models | Small CPU model plus structured evidence analysis |
| API keys | Depends on configuration | None |

The standalone profile preserves small valid result sets and shows source warnings. It never substitutes fictional papers when sources fail. Completed searches are cached for five minutes in a bounded in-memory cache. Source calls have bounded retries and a shared deadline. A process-wide budget admits 40 searches per minute, with at most three concurrent searches, and resets automatically; it is not authentication or a durable per-IP quota. Drafts can be generated repeatedly without a lifetime quota or paid calls. Model inference admits one request at a time; other requests use the evidence-analysis fallback.

## Small-model behavior

[FLAN-T5-small](https://huggingface.co/google/flan-t5-small) is a 77-million-parameter instruction model released under Apache 2.0. The demo uses its public [quantized ONNX conversion](https://huggingface.co/onnx-community/flan-t5-small-ONNX) at immutable revision `76988c16f73cadb2c2e13e2d7d85608944223105`. Encoder, decoder, and tokenizer downloads total approximately 97 MB. ONNX Runtime uses CPU execution with two threads; Torch is not installed.

The model generates concise paper and method summaries, and an overview of the selected evidence. Method, metric, dataset and limitation extraction stays tied to abstract text. Plans combine that overview with source-linked findings, missing-information checks and proposed implementation/evaluation steps derived from the evidence analysis. These are small-model drafts: fluency is limited, full papers are not read, and model text can be wrong. Numerical output absent from the supplied prompt is rejected, but this is not a complete grounding check.

Loading runs once in the background. The page reports loading/ready/failure state and refreshes when loading finishes. Generation is capped at 96 output tokens and 15 seconds, with 512 input tokens and one concurrent inference. Busy, timed-out, failed or unusable generations return extractive analysis with a visible reason. Failed loading has a retry button. Set `ACADEMICFORGE_DEMO_MODEL_ENABLED=false` to run entirely with extractive analysis and no model download.

Cold starts and download times depend on the host. [Streamlit resource limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app) are shared and can vary; local measurements are not a guarantee of hosted memory or latency.

## Deployment settings

```text
Repository: shreyshrivastava/AcademicForge
Branch: main (demo is a compatibility deployment branch)
Main file: frontend/streamlit_cloud_app.py
Python: 3.11–3.14 for the standalone profile; 3.12 recommended
Dependencies: frontend/requirements.txt beside the hosted entry point
Secrets: none required
```

The standalone profile retains Python 3.11 compatibility for existing deployments. The full local model profile requires Python 3.12–3.14. Choose Python in Streamlit deployment settings; `runtime.txt` is no longer the source of that choice. Existing deployments can keep using `demo` while it is synchronized with main. Both entry points use the same interface; the hosted entry point explicitly selects the standalone adapter.

Do not commit local `.env` files or keys. The standalone adapter downloads public ONNX weights with token authentication explicitly disabled and does not load generation credentials or call paid services, even if old generation secrets remain configured. Remove obsolete `ACADEMICFORGE_CLOUD_ENABLE_LLM`, token-limit, limiter-salt and one-use quota secrets from the deployment to avoid confusion.

The dependency file is deliberately beside the entry point: [Streamlit searches that directory before the repository root](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies). This prevents the full model `uv.lock` at the root from being selected for the standalone deployment.

## Run locally

```bash
uv venv .venv-demo --python 3.12
uv pip install --python .venv-demo/bin/python --require-hashes -r frontend/requirements.txt
.venv-demo/bin/python -m streamlit run frontend/streamlit_cloud_app.py --server.port 8503
```

The full model environment continues to use [setup.md](setup.md). `frontend/requirements.txt` is exclusively the lightweight Streamlit profile. To update its pinned, hashed dependencies, edit `requirements-cloud.in`, then run:

```bash
uv pip compile requirements-cloud.in --universal --python-version 3.11 --generate-hashes --output-file frontend/requirements.txt
```

## Verify

Check the deployed branch and commit in Streamlit's management page. Search for `rag as` with **Computer science & AI** selected, inspect the interpreted query, select evidence, open summary/guidance, generate a preview twice, and download Markdown. The page and exported plan must identify BM25 ranking, model drafts, and any extractive fallback. Source failures must show an incomplete-results warning or unavailable state.

Run the standalone contract suite with only this profile plus pytest installed:

```bash
uv pip install --python .venv-demo/bin/python 'pytest>=9.1,<10'
.venv-demo/bin/python -m pytest tests/test_streamlit_demo.py tests/test_evidence_analysis.py tests/test_small_model.py -q
```

## Retired Render services

GitHub still records failures for external Render services named `academicforge-frontend` and `academicforge-backend`. Their configuration lives in Render. Removing hosting files from the repository does not disconnect existing services.

Disable automatic deployment or disconnect the retired services in the owning Render account. Historical failures should remain accurate. The repository's Streamlit webhook and GitHub checks are separate integrations. The Vercel “Production” success beside the failed Render environments is also a separate deployment; it does not establish that the Streamlit demo is current.
