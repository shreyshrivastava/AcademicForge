FROM rocm/pytorch:rocm6.0.2_ubuntu22.04_py3.10_pytorch_2.1.2

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements-local.txt .
# Remove MLX from the requirements as it is Apple Silicon only
RUN grep -v "mlx" requirements-local.txt > requirements-container.txt
RUN pip install --no-cache-dir -r requirements-container.txt

# Copy all code
COPY . .

# Download weights into the image to satisfy the 60-second boot rule
ARG HF_TOKEN
RUN HF_TOKEN=${HF_TOKEN} python scripts/download_weights.py --llm google/gemma-2-2b-it --embed BAAI/bge-small-en-v1.5 --reranker BAAI/bge-reranker-base

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Run the entrypoint
CMD ["./start.sh"]
