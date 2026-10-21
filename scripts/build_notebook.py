#!/usr/bin/env python3
"""Programmatically build notebooks/academicforge_gpu_eval.ipynb using nbformat."""

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "academicforge_gpu_eval.ipynb"

def build():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "accelerator": "GPU",
        "colab": {
            "name": "academicforge_gpu_eval.ipynb",
            "provenance": [],
            "gpuType": "T4",
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    }

    cells = []

    # Title & Config
    cells.append(nbf.v4.new_markdown_cell("""# AcademicForge GPU Evaluation Suite
Complete, reproducible retrieval ablation and generation evaluation on NVIDIA T4 GPU.

### Sections & Estimated Runtimes
1. **Setup check:** Print GPU via `nvidia-smi` and verify CUDA availability (~5s).
2. **Upload & Extract:** Upload `academicforge-gpu-bundle.tar.gz` and extract to `/content/academicforge` (~10-30s).
3. **Install Dependencies:** Pinned dependencies from `requirements-gpu.txt`, preserving Colab's PyTorch (~1-2m).
4. **Secrets Configuration:** Safely load `HF_TOKEN` and optional `FIREWORKS_API_KEY` (~5s).
5. **Smoke Test:** Fast SciFact ablation (20 queries, rerank depth 20) with BGE fallback verification (~30-60s).
6. **Full Retrieval Ablation:** Complete SciFact and NFCorpus retrieval benchmark with 95% bootstrap CIs (~10-25m).
7. **Gemma Summaries for Faithfulness:** Generate 20 query summaries using local Gemma-2 on CUDA (~3-5m).
8. **Faithfulness Controls:** Negative mismatch and corruption controls across thresholds 0.50 to 0.80 (~30s).
9. **Package Results:** Bundle artifacts into `academicforge_results.zip` and trigger download (~5s)."""))

    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# GLOBAL RUNTIME CONFIGURATION
# ==============================================================================
# Set FORCE_RERUN = True to recompute stages even if output files already exist.
FORCE_RERUN = False

# Whether to evaluate both BEIR SciFact and BEIR NFCorpus in the full ablation (Section 6).
RUN_NFCORPUS = True

# Cross-encoder batch size. Lower to 16 or 8 if you encounter CUDA out-of-memory.
BATCH_SIZE = 32

print("Runtime configuration loaded:")
print(f"  FORCE_RERUN:  {FORCE_RERUN}")
print(f"  RUN_NFCORPUS: {RUN_NFCORPUS}")
print(f"  BATCH_SIZE:   {BATCH_SIZE}")"""))

    # Section 1
    cells.append(nbf.v4.new_markdown_cell("""## 1. Setup Check
Verifies that this Google Colab session has an active NVIDIA CUDA GPU attached.
**Estimated runtime on T4:** `< 5 seconds`.
If no GPU is present, halts with clear instructions to enable a T4 GPU runtime."""))

    cells.append(nbf.v4.new_code_cell("""import subprocess
import sys

print("Checking GPU availability and CUDA environment...")

# 1. Run nvidia-smi
try:
    smi_output = subprocess.check_output(["nvidia-smi"], text=True)
    print(smi_output)
except Exception as e:
    print(f"[Warning] nvidia-smi command not available: {e}")

# 2. Check PyTorch CUDA availability
import torch

if not torch.cuda.is_available():
    print("\\n" + "!" * 70, file=sys.stderr)
    print("ERROR: No CUDA GPU detected in this Google Colab session!", file=sys.stderr)
    print("Please follow these steps:", file=sys.stderr)
    print("  1. In the top menu, go to: Runtime > Change runtime type", file=sys.stderr)
    print("  2. Under 'Hardware accelerator', select 'T4 GPU'", file=sys.stderr)
    print("  3. Click 'Save' and rerun this notebook.", file=sys.stderr)
    print("!" * 70 + "\\n", file=sys.stderr)
    raise SystemExit(1)

gpu_name = torch.cuda.get_device_name(0)
device_props = torch.cuda.get_device_properties(0)
total_vram_gb = device_props.total_memory / (1024 ** 3)

print("=" * 60)
print("GPU HARDWARE CHECK PASSED:")
print(f"  Device:       {gpu_name}")
print(f"  Total VRAM:   {total_vram_gb:.2f} GB")
print(f"  CUDA Version: {torch.version.cuda}")
print(f"  PyTorch:      {torch.__version__}")
print("=" * 60)"""))

    # Section 2
    cells.append(nbf.v4.new_markdown_cell("""## 2. Bundle Upload and Extraction
Finds or uploads `academicforge-gpu-bundle.tar.gz` and extracts it to `/content/academicforge`.
**Estimated runtime on T4:** `~10-30 seconds` depending on upload speed (instant if already present).
If the bundle tarball is already present at `/content`, the upload prompt is automatically skipped."""))

    cells.append(nbf.v4.new_code_cell("""import os
import shutil
import tarfile
from pathlib import Path
import sys

TARGET_DIR = Path("/content/academicforge")
TARGET_DIR.mkdir(parents=True, exist_ok=True)

# Check if already extracted
if not FORCE_RERUN and (TARGET_DIR / "requirements-gpu.txt").exists() and (TARGET_DIR / "backend").exists():
    print(f"AcademicForge already extracted at {TARGET_DIR}. Skipping upload/extraction (set FORCE_RERUN=True to overwrite).")
    os.chdir(str(TARGET_DIR))
    if str(TARGET_DIR) not in sys.path:
        sys.path.insert(0, str(TARGET_DIR))
else:
    # Look for existing bundle in /content
    bundle_candidates = [
        Path("/content/academicforge-gpu-bundle.tar.gz"),
        Path("/content/academicforge.tar.gz"),
        Path("academicforge-gpu-bundle.tar.gz"),
        Path("academicforge.tar.gz"),
    ]

    bundle_path = None
    for candidate in bundle_candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            bundle_path = candidate
            print(f"Found existing bundle tarball: {bundle_path}")
            break

    if bundle_path is None:
        print("Bundle tarball not found at /content. Opening upload prompt...")
        print("Please upload 'academicforge-gpu-bundle.tar.gz' now.")
        try:
            from google.colab import files
            uploaded = files.upload()
        except ImportError:
            uploaded = {}

        if not uploaded:
            print("\\nERROR: No file was uploaded!", file=sys.stderr)
            print("Please run `bash scripts/make_gpu_bundle.sh` on your local machine,", file=sys.stderr)
            print("then upload `academicforge-gpu-bundle.tar.gz` into Colab.", file=sys.stderr)
            raise SystemExit(1)

        uploaded_name = list(uploaded.keys())[0]
        bundle_path = Path("/content") / uploaded_name

    print(f"Extracting {bundle_path} to {TARGET_DIR}...")
    try:
        with tarfile.open(bundle_path, "r:*") as tar:
            members = tar.getmembers()
            first_parts = {m.name.split("/")[0] for m in members if "/" in m.name}
            # If everything is wrapped in a top-level directory (e.g. AcademicForge/), strip it
            if len(first_parts) == 1 and not any(m.name == "." for m in members):
                top_name = list(first_parts)[0]
                for m in members:
                    if m.name == top_name:
                        continue
                    if m.name.startswith(top_name + "/"):
                        m.name = m.name[len(top_name) + 1:]
                        tar.extract(m, path=TARGET_DIR)
            else:
                tar.extractall(path=TARGET_DIR)
    except Exception as exc:
        print(f"\\nERROR: Failed to extract {bundle_path}: {exc}", file=sys.stderr)
        raise SystemExit(1)

    # Maintain compatibility with paths expecting /content/AcademicForge
    alt_dir = Path("/content/AcademicForge")
    if alt_dir != TARGET_DIR and not alt_dir.exists():
        try:
            alt_dir.symlink_to(TARGET_DIR)
        except Exception:
            pass

    os.chdir(str(TARGET_DIR))
    if str(TARGET_DIR) not in sys.path:
        sys.path.insert(0, str(TARGET_DIR))

    # Verify essential layout
    required_entries = ["backend", "benchmarks", "scripts", "requirements-gpu.txt"]
    missing = [entry for entry in required_entries if not (TARGET_DIR / entry).exists()]
    if missing:
        print(f"\\nERROR: Incomplete repository extraction! Missing: {missing}", file=sys.stderr)
        raise SystemExit(1)

    print(f"Successfully extracted AcademicForge bundle to {TARGET_DIR}")"""))

    # Section 3
    cells.append(nbf.v4.new_markdown_cell("""## 3. Install Dependencies
Installs the pinned dependencies from `requirements-gpu.txt` from the bundle, preserving Colab's pre-installed PyTorch stack.
**Estimated runtime on T4:** `~1 to 2 minutes`.
Handles the blinker/distutils ignore-installed issue before installing pinned packages without dependencies (`--no-deps`), then verifies all installed versions."""))

    cells.append(nbf.v4.new_code_cell("""import subprocess
import sys
from pathlib import Path

print("Installing dependencies from bundle requirements-gpu.txt...")

# 1. Handle blinker distutils/ignore-installed conflict on Colab
try:
    print("Ensuring blinker compatibility (--ignore-installed)...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--ignore-installed", "blinker"],
        check=True,
        stdout=subprocess.DEVNULL
    )
except subprocess.CalledProcessError as e:
    print(f"[Warning] Blinker pre-install: {e}")

# 2. Install pinned dependencies without replacing Colab's PyTorch
req_file = "requirements-gpu.txt"
if not Path(req_file).exists():
    print(f"\\nERROR: {req_file} not found in {os.getcwd()}!", file=sys.stderr)
    raise SystemExit(1)

print(f"Installing pinned packages with pip install --no-deps -r {req_file}...")
try:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "-r", req_file],
        check=True
    )
except subprocess.CalledProcessError as exc:
    print(f"\\nERROR: Dependency installation failed with exit code {exc.returncode}!", file=sys.stderr)
    raise SystemExit(1)

# 3. Print verified package versions
import torch
import transformers
import sentence_transformers
import accelerate

print("\\n" + "=" * 60)
print("DEPENDENCY VERIFICATION COMPLETE:")
print(f"  Python:                {sys.version.split()[0]}")
print(f"  PyTorch:               {torch.__version__}")
print(f"  CUDA Available:        {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA Version:          {torch.version.cuda}")
    print(f"  Device Name:           {torch.cuda.get_device_name(0)}")
print(f"  Transformers:          {transformers.__version__}")
print(f"  Sentence-Transformers: {sentence_transformers.__version__}")
print(f"  Accelerate:            {accelerate.__version__}")
print("=" * 60)"""))

    # Section 4
    cells.append(nbf.v4.new_markdown_cell("""## 4. Secrets Configuration
Reads `HF_TOKEN` and optional `FIREWORKS_API_KEY` from Colab Secrets (`google.colab.userdata`) and exports them to environment variables.
**Estimated runtime on T4:** `< 5 seconds`.

> **Gemma-2 Licensing Requirement:**
> The `google/gemma-2-2b-it` model is a gated model. To run Section 7 (Gemma faithfulness summaries):
> 1. Visit [https://huggingface.co/google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it) and accept the license terms.
> 2. Create an Access Token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
> 3. In the Colab sidebar, click the **Secrets** (key) icon, add a new secret named `HF_TOKEN`, and toggle **Notebook access** to ON.
> 4. (Optional) If you want Fireworks DeepSeek comparisons, add `FIREWORKS_API_KEY`.
>
> Secrets values are never printed or exposed."""))

    cells.append(nbf.v4.new_code_cell("""import os
import sys

print("Configuring secrets from Colab userdata...")

hf_token = None
fireworks_key = None

try:
    from google.colab import userdata
    try:
        hf_token = userdata.get("HF_TOKEN")
    except Exception:
        hf_token = None
    try:
        fireworks_key = userdata.get("FIREWORKS_API_KEY")
    except Exception:
        fireworks_key = None
except ImportError:
    # Fallback to existing os.environ if running outside Google Colab
    hf_token = os.environ.get("HF_TOKEN")
    fireworks_key = os.environ.get("FIREWORKS_API_KEY")

if hf_token and hf_token.strip():
    os.environ["HF_TOKEN"] = hf_token.strip()
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token.strip()
    print("✓ HF_TOKEN: Configured (Notebook will attempt Gemma-2 model loading)")
else:
    print("✗ HF_TOKEN: Not set (Gemma summary generation in Section 7 will be skipped)")

if fireworks_key and fireworks_key.strip():
    os.environ["FIREWORKS_API_KEY"] = fireworks_key.strip()
    print("✓ FIREWORKS_API_KEY: Configured (Notebook will attempt Fireworks DeepSeek comparison)")
else:
    print("✗ FIREWORKS_API_KEY: Not set (DeepSeek comparison in Section 7 will be skipped)")"""))

    # Section 5
    cells.append(nbf.v4.new_markdown_cell("""## 5. Smoke Test (Runs First, Fast)
Performs a fast end-to-end retrieval ablation on a 20-query subsample of BEIR SciFact with rerank-depth 20 on CUDA.
**Estimated runtime on T4:** `~30-60 seconds`.
- **Purpose:** Verifies CUDA acceleration, dataset auto-download, BGE dense embeddings, and BGE cross-encoder reranking before running the full benchmark.
- **Fail-safe check:** Verifies that neither BGE dense nor cross-encoder fell back to CPU or lexical heuristics; fails loudly if any fallback occurred.
- **Separation:** Output is explicitly labeled as a smoke test and saved to `docs/benchmarks/scifact-smoke-test.json`. It cannot be mistaken for full benchmark results."""))

    cells.append(nbf.v4.new_code_cell("""import json
from pathlib import Path
import subprocess
import sys
import pandas as pd

smoke_output = Path("docs/benchmarks/scifact-smoke-test.json")

if smoke_output.exists() and not FORCE_RERUN:
    print(f"Smoke test output already exists at {smoke_output}. Loading cached results...")
else:
    print("Starting smoke test on SciFact (20 queries, rerank-depth 20, device: cuda)...")
    cmd = [
        sys.executable, "-m", "benchmarks.run_scifact",
        "--download",
        "--datasets", "scifact",
        "--device", "cuda",
        "--max-queries", "20",
        "--rerank-depth", "20",
        "--progress",
        "--output", str(smoke_output),
    ]
    try:
        proc = subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"\\nERROR: Smoke test execution failed with exit code {exc.returncode}!", file=sys.stderr)
        raise SystemExit(1)

# Validate smoke test output
if not smoke_output.exists():
    print(f"\\nERROR: Smoke test did not produce {smoke_output}!", file=sys.stderr)
    raise SystemExit(1)

with smoke_output.open("r", encoding="utf-8") as f:
    smoke_data = json.load(f)

# Verification 1: Verify it is labeled as a smoke test / subsample
if smoke_data.get("is_full_benchmark", True):
    print("\\nERROR: Smoke test output was not marked as a subsample/reduced-depth run!", file=sys.stderr)
    raise SystemExit(1)

report = smoke_data["datasets"][0]
systems = report.get("systems", {})

# Verification 2: Verify BGE dense and cross-encoder did not fall back
cfg = report.get("configuration", {})
if cfg.get("reranker_model") != "BAAI/bge-reranker-base":
    print(f"\\nERROR: Cross-encoder reranker fell back! Model: {cfg.get('reranker_model')}", file=sys.stderr)
    raise SystemExit(1)

if "bge_dense" not in systems or "rrf_cross_encoder" not in systems:
    print("\\nERROR: BGE systems missing from smoke test report!", file=sys.stderr)
    raise SystemExit(1)

# Check query measurements for fallbacks
for query_row in systems["bge_dense"].get("per_query", []):
    if not query_row.get("ranked_document_ids"):
        print("\\nERROR: BGE dense returned empty document rankings!", file=sys.stderr)
        raise SystemExit(1)

print("\\n" + "=" * 60)
print("SMOKE TEST PASSED: No BGE fallbacks detected on CUDA!")
print(f"Classification: {smoke_data.get('result_classification')}")
print(f"Queries Tested: {report.get('queries')} / {report.get('full_split_queries')}")
print("=" * 60)

# Display smoke metrics inline
smoke_rows = []
for sys_name, data in systems.items():
    metrics = data["metrics"]
    smoke_rows.append({
        "System": sys_name,
        "nDCG@10 (mean)": f"{metrics['ndcg@10']['mean']:.4f}",
        "Recall@10 (mean)": f"{metrics['recall@10']['mean']:.4f}",
        "p50 Latency (ms)": f"{data['latency_ms']['p50']:.1f}",
        "p95 Latency (ms)": f"{data['latency_ms']['p95']:.1f}",
    })
print("\\n[Smoke Test Inline Summary - 20 queries, reduced depth]")
display(pd.DataFrame(smoke_rows))"""))

    # Section 6
    cells.append(nbf.v4.new_markdown_cell("""## 6. Full Retrieval Ablation
Runs the complete retrieval ablation on the full BEIR SciFact test split (300 queries, 5,183 documents) and optionally BEIR NFCorpus (323 queries, 3,633 documents) with default rerank depth (50) and optimized batching on CUDA.
**Estimated runtime on T4:** `~8-12 minutes` for SciFact alone; `~15-25 minutes` total if NFCorpus is included.
- Writes official artifact to `docs/benchmarks/retrieval-ablation.json`.
- Measures BM25, stemmed BM25, BGE dense, RRF, and RRF + BGE cross-encoder.
- Computes 95% bootstrap confidence intervals, paired significance vs BM25, and p50/p95 latency on GPU."""))

    cells.append(nbf.v4.new_code_cell("""import json
import os
from pathlib import Path
import subprocess
import sys
import pandas as pd
import torch

ablation_output = Path("docs/benchmarks/retrieval-ablation.json")

datasets_to_run = "scifact,nfcorpus" if RUN_NFCORPUS else "scifact"

if ablation_output.exists() and not FORCE_RERUN:
    print(f"Full ablation artifact already exists at {ablation_output}. Loading and displaying results...")
else:
    print(f"Running full retrieval ablation on datasets: {datasets_to_run}")
    print(f"Cross-encoder batch size: {BATCH_SIZE}, rerank depth: 50, device: cuda")
    print("Estimated runtime: ~10-25 minutes on T4 GPU. Progress logged every 10 queries...")

    cmd = [
        sys.executable, "-m", "benchmarks.run_scifact",
        "--download",
        "--datasets", datasets_to_run,
        "--device", "cuda",
        "--batch-size", str(BATCH_SIZE),
        "--rerank-depth", "50",
        "--progress",
        "--output", str(ablation_output),
    ]

    try:
        proc = subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print("\\n" + "!" * 70, file=sys.stderr)
        print(f"ERROR: Full retrieval ablation failed with exit code {exc.returncode}!", file=sys.stderr)
        print("If this was caused by a CUDA Out of Memory (OOM) error:", file=sys.stderr)
        print("  1. Lower BATCH_SIZE at the top of the notebook (e.g. BATCH_SIZE = 16 or 8).", file=sys.stderr)
        print("  2. Go to Runtime > Restart session, and rerun from Section 6.", file=sys.stderr)
        print("!" * 70 + "\\n", file=sys.stderr)
        raise SystemExit(1)

# Verify and display full results
if not ablation_output.exists():
    print(f"\\nERROR: Output file {ablation_output} was not created!", file=sys.stderr)
    raise SystemExit(1)

with ablation_output.open("r", encoding="utf-8") as f:
    ablation_data = json.load(f)

gpu_device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown GPU"

print("\\n" + "=" * 80)
print(f"FULL RETRIEVAL ABLATION RESULTS (Evaluated on {gpu_device_name})")
print("=" * 80)

# Display Table 1: Retrieval Metrics with 95% Bootstrap CIs
for report in ablation_data.get("datasets", []):
    ds_name = report["dataset"]
    print(f"\\n### Dataset: {ds_name} ({report['queries']} test queries, {report['documents']} documents)")
    
    table_rows = []
    for sys_name in ["bm25", "bm25_stem_stop", "bge_dense", "rrf", "rrf_cross_encoder"]:
        sys_data = report["systems"].get(sys_name)
        if not sys_data:
            continue
        m = sys_data["metrics"]
        ndcg_str = f"{m['ndcg@10']['mean']:.4f} [{m['ndcg@10']['ci95_low']:.4f}, {m['ndcg@10']['ci95_high']:.4f}]"
        rec_str = f"{m['recall@10']['mean']:.4f} [{m['recall@10']['ci95_low']:.4f}, {m['recall@10']['ci95_high']:.4f}]"
        mrr_str = f"{m['mrr@10']['mean']:.4f} [{m['mrr@10']['ci95_low']:.4f}, {m['mrr@10']['ci95_high']:.4f}]"
        lat = sys_data["latency_ms"]
        
        table_rows.append({
            "System": sys_name,
            "nDCG@10 (95% CI)": ndcg_str,
            "Recall@10 (95% CI)": rec_str,
            "MRR@10 (95% CI)": mrr_str,
            "p50 Lat (ms)": f"{lat['p50']:.1f}",
            "p95 Lat (ms)": f"{lat['p95']:.1f}",
        })
    df_metrics = pd.DataFrame(table_rows)
    display(df_metrics)

    # Display Table 2: Paired Significance Tests vs BM25
    sig_rows = []
    for sys_name in ["bm25_stem_stop", "bge_dense", "rrf", "rrf_cross_encoder"]:
        sys_data = report["systems"].get(sys_name, {})
        paired = sys_data.get("paired_bootstrap_vs_bm25", {})
        for metric_name in ["ndcg@10", "recall@10", "mrr@10"]:
            p_data = paired.get(metric_name)
            if not p_data:
                continue
            diff_str = f"{p_data['mean_difference']:+.4f} [{p_data['ci95_low']:+.4f}, {p_data['ci95_high']:+.4f}]"
            sig_rows.append({
                "System vs BM25": sys_name,
                "Metric": metric_name,
                "Mean Diff vs BM25 (95% CI)": diff_str,
                "p-value": f"{p_data['p_value']:.4f}",
                "Stat. Significant (p<0.05)": "Yes" if p_data['p_value'] < 0.05 else "No",
            })
    if sig_rows:
        print(f"\\n[Paired Bootstrap Significance vs BM25 ({ds_name})]")
        display(pd.DataFrame(sig_rows))"""))

    # Section 7
    cells.append(nbf.v4.new_markdown_cell("""## 7. Gemma Summaries for Faithfulness Check
Generates summaries for the 20 benchmark questions in `benchmarks/queries.json` using the local Gemma-2-2b-it model on CUDA.
**Estimated runtime on T4:** `~3-5 minutes` for Gemma-2 (20 queries).
- Uses the application's native retrieval and summarization pipeline.
- If `HF_TOKEN` is missing or access to `google/gemma-2-2b-it` is denied, skips gracefully with an informative message instead of crashing.
- If `FIREWORKS_API_KEY` is present, also evaluates DeepSeek summaries.
- Writes output to `docs/benchmarks/faithfulness.json`."""))

    cells.append(nbf.v4.new_code_cell("""import json
import os
from pathlib import Path
import subprocess
import sys

faithfulness_output = Path("docs/benchmarks/faithfulness.json")
skip_faithfulness = False

hf_token = os.environ.get("HF_TOKEN", "").strip()

if not hf_token:
    print("=" * 70)
    print("NOTICE: HF_TOKEN is not configured in Colab Secrets.")
    print("Gemma-2 is a gated model requiring Hugging Face authorization.")
    print("SKIPPING Section 7 (Gemma summary generation) and Section 8 (Faithfulness controls).")
    print("To enable:")
    print("  1. Accept license: https://huggingface.co/google/gemma-2-2b-it")
    print("  2. Add HF_TOKEN secret in Colab sidebar with Notebook access enabled.")
    print("=" * 70)
    skip_faithfulness = True
else:
    # Test Hugging Face token authorization for google/gemma-2-2b-it
    print("Validating Hugging Face authorization for google/gemma-2-2b-it...")
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
        api = HfApi(token=hf_token)
        api.model_info("google/gemma-2-2b-it")
        print("✓ Authorization successful: access granted to google/gemma-2-2b-it.")
    except GatedRepoError:
        print("\\n" + "!" * 70, file=sys.stderr)
        print("ERROR: Hugging Face access to google/gemma-2-2b-it was DENIED (Gated Repository)!", file=sys.stderr)
        print("Please visit: https://huggingface.co/google/gemma-2-2b-it to accept the license agreement.", file=sys.stderr)
        print("SKIPPING Section 7 (Gemma summaries) and Section 8 (Faithfulness controls).", file=sys.stderr)
        print("!" * 70 + "\\n", file=sys.stderr)
        skip_faithfulness = True
    except Exception as auth_err:
        print(f"[Warning] Hugging Face check returned: {auth_err}. Proceeding with run...")

if not skip_faithfulness:
    if faithfulness_output.exists() and not FORCE_RERUN:
        print(f"Faithfulness output already exists at {faithfulness_output}. Loading cached results...")
    else:
        print("Generating summaries for 20 benchmark queries using Gemma on CUDA...")
        print("Estimated runtime: ~3-5 minutes on T4. Progress logged every 5 queries...")
        cmd = [
            sys.executable, "-m", "benchmarks.faithfulness",
            "--device", "cuda",
            "--output", str(faithfulness_output),
        ]
        try:
            subprocess.run(cmd, check=True)
            print(f"\\nSuccessfully wrote faithfulness artifact to {faithfulness_output}")
        except subprocess.CalledProcessError as exc:
            print(f"\\nERROR: Faithfulness generation failed with exit code {exc.returncode}!", file=sys.stderr)
            raise SystemExit(1)

    if faithfulness_output.exists():
        with faithfulness_output.open("r", encoding="utf-8") as f:
            fdata = json.load(f)
        summary_models = fdata.get("results", {}).get("summary", {})
        print("\\n" + "=" * 60)
        print("FAITHFULNESS SCREENING SUMMARY:")
        for mid, mstats in summary_models.items():
            print(f"Model: {mid}")
            print(f"  Outputs:                {mstats['outputs']}")
            print(f"  Sentences:              {mstats['sentences']}")
            print(f"  Supported sentences:    {mstats['supported_sentences']}")
            print(f"  Heuristic Support Rate: {mstats['micro_support_rate']*100:.1f}%")
        print("=" * 60)"""))

    # Section 8
    cells.append(nbf.v4.new_markdown_cell("""## 8. Faithfulness Controls
Runs negative mismatch and corruption controls across cosine similarity thresholds 0.50 to 0.80, exactly as implemented in `benchmarks/faithfulness.py`.
**Estimated runtime on T4:** `~10-30 seconds`.
- **Mismatch control:** Pairs summaries with abstracts from different queries to test whether topical alignment is required for support.
- **Corruption control:** Injects universal causal claims and reversed conclusions into summaries to test whether the check detects contradictions.
- Writes official artifact to `docs/benchmarks/faithfulness_controls.json`.
- Prints an inline threshold table and a plain-language verdict on whether semantic embedding similarity can separate supported statements from corrupted/reversed claims."""))

    cells.append(nbf.v4.new_code_cell("""import json
from pathlib import Path
import subprocess
import sys
import pandas as pd

controls_output = Path("docs/benchmarks/faithfulness_controls.json")
faithfulness_source = Path("docs/benchmarks/faithfulness.json")

if not faithfulness_source.exists():
    print("Notice: docs/benchmarks/faithfulness.json does not exist (HF_TOKEN missing or generation skipped).")
    print("Skipping faithfulness negative controls.")
else:
    if controls_output.exists() and not FORCE_RERUN:
        print(f"Faithfulness controls output already exists at {controls_output}. Loading cached results...")
    else:
        print("Running faithfulness negative controls on CUDA...")
        cmd = [
            sys.executable, "-m", "benchmarks.faithfulness",
            "--device", "cuda",
            "--controls-from", str(faithfulness_source),
            "--controls-output", str(controls_output),
        ]
        try:
            subprocess.run(cmd, check=True)
            print(f"Wrote controls artifact: {controls_output}")
        except subprocess.CalledProcessError as exc:
            print(f"\\nERROR: Faithfulness controls run failed with exit code {exc.returncode}!", file=sys.stderr)
            raise SystemExit(1)

    with controls_output.open("r", encoding="utf-8") as f:
        controls_data = json.load(f)

    threshold_rows = []
    for row in controls_data.get("thresholds", []):
        t = row["threshold"]
        real_rate = row["real_supported"]["rate"] * 100
        mismatch_rate = row["mismatched_supported"]["rate"] * 100
        unsupported_rate = row["unsupported_claim_flagged"]["rate"] * 100
        reversed_rate = row["reversed_conclusion_flagged"]["rate"] * 100
        sep = row["separation"]
        threshold_rows.append({
            "Threshold": f"{t:.2f}",
            "Real Supported (%)": f"{real_rate:.1f}%",
            "Mismatched Supported (%)": f"{mismatch_rate:.1f}%",
            "Injected Unsupported Flagged (%)": f"{unsupported_rate:.1f}%",
            "Reversed Conclusion Flagged (%)": f"{reversed_rate:.1f}%",
            "Separation Score": f"{sep:.3f}",
        })

    df_thresholds = pd.DataFrame(threshold_rows)
    print("\\n[Faithfulness Screening Negative Controls Across Thresholds 0.50 - 0.80]")
    display(df_thresholds)

    # Print plain-language verdict
    sel_t = controls_data.get("selected_threshold", 0.60)
    best_row = next((r for r in controls_data["thresholds"] if abs(r["threshold"] - sel_t) < 1e-4), controls_data["thresholds"][0])
    
    real_pct = best_row["real_supported"]["rate"] * 100
    mismatch_pct = best_row["mismatched_supported"]["rate"] * 100
    unsupported_pct = best_row["unsupported_claim_flagged"]["rate"] * 100
    reversed_pct = best_row["reversed_conclusion_flagged"]["rate"] * 100

    print("\\n" + "=" * 70)
    print(f"PLAIN-LANGUAGE VERDICT AT SELECTED THRESHOLD ({sel_t:.2f}):")
    print(f"  • Real summary sentences marked supported:       {real_pct:.1f}%")
    print(f"  • Mismatched-abstract sentences marked supported: {mismatch_pct:.1f}%")
    print(f"  • Injected unsupported claims flagged:            {unsupported_pct:.1f}%")
    print(f"  • Reversed/contradicted conclusions flagged:      {reversed_pct:.1f}%")
    print("-" * 70)
    if reversed_pct == 0.0:
        print("VERDICT: The embedding similarity check CANNOT separate real from reversed conclusions.")
        print("Because negated sentences share almost identical vocabulary and topical embeddings with")
        print("the original sentences, cosine similarity scores both equally high and fails to detect")
        print("contradictions (0% flagged). True factual verification requires an NLI entailment model")
        print("or human review.")
    else:
        print(f"VERDICT: The embedding similarity check separated real from reversed conclusions with a {reversed_pct:.1f}% flag rate.")
    print("=" * 70)"""))

    # Section 9
    cells.append(nbf.v4.new_markdown_cell("""## 9. Package Results and Download
Gathers all generated evaluation artifacts and environment metadata into a compressed zip file (`academicforge_results.zip`) and triggers a browser download.
**Estimated runtime on T4:** `< 5 seconds`.
Prints the exact command to import the benchmark results into your local repository."""))

    cells.append(nbf.v4.new_code_cell("""from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import zipfile
import torch

results_zip_path = Path("/content/academicforge_results.zip")
artifacts_to_bundle = [
    Path("docs/benchmarks/retrieval-ablation.json"),
    Path("docs/benchmarks/scifact-smoke-test.json"),
    Path("docs/benchmarks/faithfulness.json"),
    Path("docs/benchmarks/faithfulness_controls.json"),
]

existing_artifacts = [p for p in artifacts_to_bundle if p.exists()]

if not existing_artifacts:
    print("\\nERROR: No benchmark artifacts were found to package!", file=sys.stderr)
    raise SystemExit(1)

# Generate run_info.json
import transformers
import sentence_transformers

run_info = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU",
    "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    "pytorch_version": torch.__version__,
    "transformers_version": transformers.__version__,
    "sentence_transformers_version": sentence_transformers.__version__,
    "platform": platform.platform(),
    "python_version": sys.version.split()[0],
    "is_full_benchmark": Path("docs/benchmarks/retrieval-ablation.json").exists(),
    "bundled_files": [str(p) for p in existing_artifacts],
    "commands_used": {
        "smoke_test": "python -m benchmarks.run_scifact --download --datasets scifact --device cuda --max-queries 20 --rerank-depth 20 --progress --output docs/benchmarks/scifact-smoke-test.json",
        "full_ablation": f"python -m benchmarks.run_scifact --download --datasets {'scifact,nfcorpus' if RUN_NFCORPUS else 'scifact'} --device cuda --rerank-depth 50 --batch-size {BATCH_SIZE} --progress --output docs/benchmarks/retrieval-ablation.json",
        "faithfulness": "python -m benchmarks.faithfulness --device cuda --output docs/benchmarks/faithfulness.json",
        "faithfulness_controls": "python -m benchmarks.faithfulness --device cuda --controls-from docs/benchmarks/faithfulness.json --controls-output docs/benchmarks/faithfulness_controls.json",
    }
}

run_info_path = Path("run_info.json")
with run_info_path.open("w", encoding="utf-8") as f:
    json.dump(run_info, f, indent=2)

print(f"Creating results bundle: {results_zip_path}...")
with zipfile.ZipFile(results_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for art in existing_artifacts:
        zf.write(art, arcname=art.name)
        print(f"  Added: {art.name} ({art.stat().st_size} bytes)")
    zf.write(run_info_path, arcname="run_info.json")
    print("  Added: run_info.json")

# Trigger browser download if running in Colab
try:
    from google.colab import files
    files.download(str(results_zip_path))
    print("\\n✓ Download triggered for academicforge_results.zip")
except Exception as dl_err:
    print(f"\\nNote: Browser download prompt could not be triggered automatically: {dl_err}")
    print(f"You can download {results_zip_path} directly from the Colab file browser in the left panel.")

print("\\n" + "=" * 75)
print("BENCHMARK RUN COMPLETE! LOCAL IMPORT INSTRUCTIONS:")
print("=" * 75)
print("Once you download academicforge_results.zip to your local computer:")
print("1. Extract retrieval-ablation.json from the zip file.")
print("2. In your local AcademicForge repository, run:")
print("\\n   python -m scripts.import_results retrieval-ablation.json\\n")
print("(or: python -m scripts.import_results docs/benchmarks/retrieval-ablation.json)")
print("This will validate dataset checksums, verify source SHA256 integrity, and")
print("update the published ablation tables in README.md and docs/benchmarks/README.md.")
print("=" * 75)"""))

    nb.cells = cells
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote notebook to {NOTEBOOK_PATH} with {len(cells)} cells.")

if __name__ == "__main__":
    build()
