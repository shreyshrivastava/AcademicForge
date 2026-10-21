#!/usr/bin/env bash
set -euo pipefail
export COPYFILE_DISABLE=1

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

git_head=$(git -C "$root" rev-parse HEAD 2>/dev/null || echo "unknown")
printf "%s\n" "$git_head" > "$stage/AcademicForge/.git_head"

# Initialize a clean standalone git repository in the bundle so git rev-parse HEAD succeeds
if command -v git >/dev/null 2>&1; then
  git -C "$stage/AcademicForge" init -q -b benchmark-evidence
  git -C "$stage/AcademicForge" config user.email "bundle@academicforge.local"
  git -C "$stage/AcademicForge" config user.name "AcademicForge Bundle"
  git -C "$stage/AcademicForge" add -A
  GIT_COMMITTER_DATE="2026-10-22T00:00:00" GIT_AUTHOR_DATE="2026-10-22T00:00:00" \
    git -C "$stage/AcademicForge" commit -q -m "Bundle export at $git_head" || true
fi

cat > "$stage/AcademicForge/RUN_ON_GPU.md" <<'RUNEOF'
# Run the retrieval ablation on a GPU machine

This bundle contains no API keys, virtual environment, or model cache.
The run downloads public BEIR datasets and Hugging Face model weights on first use.

## Google Colab T4 Notebook (Recommended)

1. Open `notebooks/academicforge_gpu_eval.ipynb` in Google Colab.
2. Select Runtime > Change runtime type > T4 GPU.
3. If running Gemma faithfulness evaluation:
   - Accept the license at https://huggingface.co/google/gemma-2-2b-it
   - Add your Hugging Face token in Colab Secrets (the key icon on the left sidebar) as `HF_TOKEN`.
   - Optional: Add `FIREWORKS_API_KEY` for Fireworks DeepSeek comparisons.
4. Run all cells. When prompted in Cell 2, upload `academicforge-gpu-bundle.tar.gz`.
5. When complete, download `academicforge_results.zip` and import:

```bash
python -m scripts.import_results docs/benchmarks/retrieval-ablation.json
```

## CLI / ROCm VM

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
python -m scripts.import_results docs/benchmarks/retrieval-ablation.json
```
RUNEOF

tar -C "$stage" -czf "$output" AcademicForge
printf 'Created %s\n' "$output"
