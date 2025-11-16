FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/cache \
    TRANSFORMERS_CACHE=/app/cache \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0" \
    CUDA_LAUNCH_BLOCKING=0 \
    TORCH_USE_CUDA_DSA=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# System dependencies - INCLUDING GIT
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git \
    wget curl build-essential \
    libgl1-mesa-glx libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip & core packaging tools
RUN pip install --upgrade pip setuptools wheel

WORKDIR /app

# Install PyTorch with CUDA 12.8
RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Verify CUDA / PyTorch
RUN python3 - <<'EOF'
import torch
print('='*60)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Runtime: {torch.version.cuda}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
print('='*60)
EOF

# App dependencies and source files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY model_manager.py .
COPY comfyui_client.py .
COPY infinitetalk_hybrid.py .
COPY infinitetalk_wrapper.py .
COPY templates/ /app/templates/
COPY comfyui_workflows/ /app/comfyui_workflows/

# Create directories
RUN mkdir -p /app/models /app/outputs /app/cache /app/templates

# Launch the FastAPI app
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
