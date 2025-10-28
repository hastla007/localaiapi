# =====================================================
# Optimized for RTX 5070 Ti (Blackwell, sm_120)
# CUDA 12.8.0 + PyTorch Nightly + FastAPI / Uvicorn app
# =====================================================
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0" \
    CUDA_LAUNCH_BLOCKING=0 \
    TORCH_USE_CUDA_DSA=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git wget curl build-essential \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip & core packaging tools
RUN pip install --upgrade pip setuptools wheel

WORKDIR /app

# Install latest nightly PyTorch with CUDA 12.8 (Blackwell GPU support)
RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Pre-download Flux low VRAM components
RUN mkdir -p /app/models/flux/unet \
    /app/models/flux/vae \
    /app/models/flux/clip

# Ensure huggingface-cli is available inside the image
RUN pip install --no-cache-dir --upgrade huggingface-hub

# Download Flux UNet weights
RUN huggingface-cli download black-forest-labs/FLUX.1-dev \
    flux1-dev.safetensors \
    --local-dir /app/models/flux/unet \
    --local-dir-use-symlinks False

# Download Flux VAE weights
RUN huggingface-cli download black-forest-labs/FLUX.1-dev \
    ae.safetensors \
    --local-dir /app/models/flux/vae \
    --local-dir-use-symlinks False

# Download CLIP text encoders (fp8 + CLIP-L)
RUN huggingface-cli download comfyanonymous/flux_text_encoders \
    clip_l.safetensors \
    --local-dir /app/models/flux/clip \
    --local-dir-use-symlinks False

RUN huggingface-cli download comfyanonymous/flux_text_encoders \
    t5xxl_fp8_e4m3fn.safetensors \
    --local-dir /app/models/flux/clip \
    --local-dir-use-symlinks False


# Verify installation
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

# App dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY model_manager.py .
COPY templates/ /app/templates/

RUN mkdir -p /app/models /app/outputs /app/cache /app/templates

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
