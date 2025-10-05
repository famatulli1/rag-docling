# Use Python 3.12 slim image
FROM python:3.12-slim

# Install system dependencies for docling (OCR support)
# Note: SmolVLM-256M model (~500MB) will be downloaded on first use
# Ensure sufficient disk space (minimum 5GB recommended)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files
COPY services/rag_server/pyproject.toml ./

# Install dependencies using uv with CUDA-enabled PyTorch for GPU acceleration
# Use CUDA 11.8 compatible PyTorch for NVIDIA GPU support
RUN uv sync --extra-index-url https://download.pytorch.org/whl/cu118 --index-strategy unsafe-best-match

# Copy application code
COPY services/rag_server/ ./

# Expose port
EXPOSE 8001

# Run using the pre-built virtual environment directly
CMD [".venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
