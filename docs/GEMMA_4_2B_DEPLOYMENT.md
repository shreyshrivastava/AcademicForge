# Gemma 4 2B Deployment Guide

This guide details the deployment of the `google/gemma-4-2b-it` model on an AMD GPU target using the ROCm environment. This is the fallback/local path when the primary Fireworks API is not in use.

## Why Gemma 4 2B?

- **Smallest Gemma 4 Variant**: Lowest VRAM footprint compared to the 9B and 31B variants.
- **Fastest Startup**: Requires minimal time to load weights into VRAM.
- **ROCm Testing**: The best candidate for verifying PyTorch / Transformers execution on AMD hardware due to lower failure rates from OOM errors.

## AMD ROCm Setup

When deploying to the AMD environment, ensure the correct version of PyTorch for ROCm is installed.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
pip install transformers accelerate bitsandbytes
```

To run the backend with the Transformers provider and 4-bit quantization:
```bash
export LOCAL_LLM_PROVIDER=transformers
export LOCAL_LLM_LOAD_IN_4BIT=true
export BACKEND_MODE=amd
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

## Lazy Loading and Resident Memory

The backend leverages `functools.lru_cache(maxsize=2)` on the `_load_transformers_model` initialization function. 
- **Lazy Loading**: The model weights (`google/gemma-4-2b-it`) are not loaded into VRAM until the *first* request (Summary, Guidance, or Research Plan) hits the API.
- **Resident Memory**: Once the first request finishes, the loaded PyTorch model and tokenizer remain cached in the application's RAM/VRAM indefinitely, making all subsequent requests start instantly.

## Telemetry and Performance Measurement

The backend actively monitors and logs the model lifecycle. In the terminal running your FastAPI app, look for standard log outputs:

```
INFO backend.llm Model 'google/gemma-4-2b-it' loaded in 4.30s. Inference completed in 1.15s.
```

- **Load Time**: Measures the duration of `AutoModelForCausalLM.from_pretrained()`. This will be `0.00s` on subsequent calls due to caching.
- **Inference Time**: Measures the actual forward-pass generation loop through the ROCm hardware.
