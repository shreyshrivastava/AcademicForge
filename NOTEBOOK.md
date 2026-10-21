# AcademicForge GPU Evaluation Guide

This guide describes how to run the AcademicForge retrieval ablation and faithfulness evaluation on a free Google Colab T4 GPU runtime.

---

## 1. Build the GPU Bundle Tarball

From the root of your local `AcademicForge` repository on branch `benchmark-evidence`:

```bash
bash scripts/make_gpu_bundle.sh
```

This creates `academicforge-gpu-bundle.tar.gz` in the repository root. The bundle contains the application code, benchmark scripts, requirements, and evaluation metadata, excluding local virtual environments and caches.

---

## 2. Open Notebook in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Select **Upload** and upload `notebooks/academicforge_gpu_eval.ipynb` from this repository.
3. Ensure hardware acceleration is enabled:
   - In the top menu, go to **Runtime > Change runtime type**.
   - Select **T4 GPU** under Hardware accelerator.
   - Click **Save**.

---

## 3. Configure Hugging Face Secret

The local generation and faithfulness checks evaluate `google/gemma-2-2b-it`, which is a gated model on Hugging Face:

1. Accept the model license terms at [https://huggingface.co/google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it).
2. Generate an Access Token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. In Google Colab, click the **Secrets** (key) icon on the left sidebar.
4. Click **Add new secret**:
   - **Name:** `HF_TOKEN`
   - **Value:** `<your-huggingface-token>`
   - Toggle **Notebook access** to **ON**.
5. *(Optional)* If evaluating Fireworks DeepSeek comparisons, add `FIREWORKS_API_KEY` under Secrets as well.

> **Note:** If `HF_TOKEN` is omitted, the notebook will run the complete retrieval ablation and smoke test, but will gracefully skip the Gemma generation and faithfulness controls sections without crashing.

---

## 4. Run All Cells

Select **Runtime > Run all** (or press `Ctrl+F9` / `Cmd+F9`):

- **Cell 1 (Setup Check):** Verifies the NVIDIA T4 GPU and PyTorch CUDA environment (< 5s).
- **Cell 2 (Bundle Upload):** When prompted, upload `academicforge-gpu-bundle.tar.gz`. If the tarball is already present at `/content/academicforge-gpu-bundle.tar.gz`, the prompt is skipped automatically (~30s).
- **Cell 3 (Install):** Installs pinned dependencies from `requirements-gpu.txt` with `--no-deps`, resolving the Colab blinker distutils issue (~1-2m).
- **Cell 4 (Secrets):** Reads `HF_TOKEN` and optional `FIREWORKS_API_KEY` from Colab userdata (< 5s).
- **Cell 5 (Smoke Test):** Fast 20-query SciFact ablation on CUDA with rerank-depth 20; checks for BGE dense/cross-encoder fallbacks (~30-60s).
- **Cell 6 (Full Ablation):** Complete SciFact (300 queries) and NFCorpus (323 queries) retrieval ablation on CUDA; outputs `docs/benchmarks/retrieval-ablation.json` and renders inline metrics tables with 95% bootstrap CIs and paired significance tests (~10-25m).
- **Cell 7 (Gemma Summaries):** Generates summaries for 20 queries on CUDA; outputs `docs/benchmarks/faithfulness.json` (~3-5m).
- **Cell 8 (Faithfulness Controls):** Runs mismatch and corruption negative controls across thresholds 0.50 to 0.80; outputs `docs/benchmarks/faithfulness_controls.json` and prints a plain-language verdict on contradiction detection (~30s).
- **Cell 9 (Package Results):** Packages all generated JSON artifacts and `run_info.json` into `academicforge_results.zip` and triggers automatic browser download (< 5s).

---

## 5. Import Results Locally

Once `academicforge_results.zip` is downloaded to your local machine:

1. Extract `retrieval-ablation.json` from the zip archive into your repository (e.g. into `docs/benchmarks/retrieval-ablation.json` or current working directory).
2. In your local repository, run:

```bash
python -m scripts.import_results retrieval-ablation.json
```

or:

```bash
python -m scripts.import_results docs/benchmarks/retrieval-ablation.json
```

This script validates dataset checksums, verifies source code SHA256 integrity against your checkout, and automatically refreshes the generated ablation tables in `README.md` and `docs/benchmarks/README.md`.
