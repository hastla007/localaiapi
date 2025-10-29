# =====================================================
# Optimized for RTX 5070 Ti (Blackwell, sm_120)
# CUDA 12.8.0 + PyTorch Stable + FastAPI / Uvicorn app
# =====================================================
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Accept HuggingFace token as build argument
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# DEBUG: Verify token is set (shows only first 10 chars for security)
RUN echo "HF_TOKEN received: ${HF_TOKEN:0:10}..." && \
    if [ -z "$HF_TOKEN" ]; then echo "ERROR: HF_TOKEN is empty!"; exit 1; fi

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0" \
    CUDA_LAUNCH_BLOCKING=0 \
    TORCH_USE_CUDA_DSA=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# -----------------------------------------------------
# System dependencies
# -----------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git wget curl build-essential \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip & core packaging tools
RUN pip install --upgrade pip setuptools wheel

WORKDIR /app

# -----------------------------------------------------
# Install PyTorch with CUDA 12.4 (stable, Blackwell compatible)
# -----------------------------------------------------
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# -----------------------------------------------------
# Prepare model directories
# -----------------------------------------------------
RUN mkdir -p /app/models/flux/unet \
    /app/models/flux/vae \
    /app/models/flux/clip

# -----------------------------------------------------
# Install Hugging Face Hub
# -----------------------------------------------------
RUN pip install --no-cache-dir --upgrade huggingface-hub

# -----------------------------------------------------
# Download model weights using Python (more reliable than CLI)
# -----------------------------------------------------

# Download Flux UNet weights (with authentication)
RUN python3 -c "from huggingface_hub import hf_hub_download; import os; hf_hub_download(repo_id='black-forest-labs/FLUX.1-dev', filename='flux1-dev.safetensors', local_dir='/app/models/flux/unet', local_dir_use_symlinks=False, token=os.getenv('HF_TOKEN'))"

# Download Flux VAE weights (with authentication)
RUN python3 -c "from huggingface_hub import hf_hub_download; import os; hf_hub_download(repo_id='black-forest-labs/FLUX.1-dev', filename='ae.safetensors', local_dir='/app/models/flux/vae', local_dir_use_symlinks=False, token=os.getenv('HF_TOKEN'))"

# Download CLIP text encoders (no auth needed)
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='comfyanonymous/flux_text_encoders', filename='clip_l.safetensors', local_dir='/app/models/flux/clip', local_dir_use_symlinks=False)"

RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='comfyanonymous/flux_text_encoders', filename='t5xxl_fp8_e4m3fn.safetensors', local_dir='/app/models/flux/clip', local_dir_use_symlinks=False)"


# -----------------------------------------------------
# Verify CUDA / PyTorch
# -----------------------------------------------------
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

# -----------------------------------------------------
# App dependencies and source files
# -----------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY model_manager.py .
COPY templates/ /app/templates/

RUN mkdir -p /app/models /app/outputs /app/cache /app/templates

# -----------------------------------------------------
# Launch the FastAPI app
# -----------------------------------------------------
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
