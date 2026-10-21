# AMD ROCm setup

Use the [GPU VM setup guide](setup.md#gpu-vms) with a verified ROCm PyTorch 2.12.x image. After activating its Python environment, run:

```bash
python -m scripts.install_gpu rocm
cp .env.example .env
```

Set `LOCAL_LLM_PROVIDER=transformers`. Add `HF_TOKEN` if using gated Gemma weights. For a notebook proxy, follow the [network access instructions](setup.md#network-access), then run `./start.sh`.

The `/version` endpoint should report `accelerator: rocm`. `retrieval_device: cuda` is expected: PyTorch exposes AMD GPUs through its CUDA-compatible device API. Record this response and perform a real search and generation on the deployment image; unit tests do not verify GPU compatibility.
