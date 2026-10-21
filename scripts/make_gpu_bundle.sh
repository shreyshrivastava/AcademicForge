#!/usr/bin/env bash
set -euo pipefail

# Package the working tree for an offline handoff. Git metadata, environments,
# caches, and ignored local credentials are deliberately excluded.
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output=${1:-"$root/academicforge-gpu-bundle.tar.gz"}
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT

mkdir -p "$stage/AcademicForge"
git -C "$root" ls-files -co --exclude-standard -z |
  tar --null --files-from=- --no-recursion -cf - -C "$root" |
  tar -xf - -C "$stage/AcademicForge"

cat > "$stage/AcademicForge/RUN_ON_GPU.md" <<'EOF'
# Run the retrieval ablation on a GPU machine

This bundle contains no API keys, `.git` directory, virtual environment, or model cache.
The run downloads public BEIR datasets and Hugging Face model weights on first use.

## Google Colab or Kaggle T4

1. Enable a T4 GPU in the notebook settings, then upload `academicforge-gpu-bundle.tar.gz`.
2. Run:

```bash
tar -xzf academicforge-gpu-bundle.tar.gz
cd AcademicForge
python -m pip install --upgrade pip
python -m pip install --upgrade --force-reinstall 'torch==2.12.1' --index-url https://download.pytorch.org/whl/cu126
python -m pip install --no-deps -r requirements-gpu.txt
python -m scripts.install_gpu cuda
python -m benchmarks.run_scifact --download --datasets scifact,nfcorpus --device cuda --rerank-depth 50 --batch-size 32 --progress --output docs/benchmarks/retrieval-ablation.json
```

3. Download `docs/benchmarks/retrieval-ablation.json` from the notebook file browser.

For a smoke test only, append `--max-queries 20 --rerank-depth 10`. That JSON is automatically labeled `subsample, reduced-depth`; it is not a benchmark result.

## AMD ROCm VM

From your local machine:

```bash
scp academicforge-gpu-bundle.tar.gz USER@ROCM_VM:/path/to/work/
```

On a ROCm VM with a supported PyTorch 2.12.x ROCm installation already available:

```bash
cd /path/to/work
tar -xzf academicforge-gpu-bundle.tar.gz
cd AcademicForge
python -m pip install --upgrade pip
python -m pip install --no-deps -r requirements-gpu.txt
python -m scripts.install_gpu rocm
python -m benchmarks.run_scifact --download --datasets scifact,nfcorpus --device cuda --rerank-depth 50 --batch-size 32 --progress --output docs/benchmarks/retrieval-ablation.json
```

Copy the JSON back, then validate and import it from the same source checkout:

```bash
python -m scripts.import_results /path/to/retrieval-ablation.json
```
EOF

tar -C "$stage" -czf "$output" AcademicForge
printf 'Created %s\n' "$output"
