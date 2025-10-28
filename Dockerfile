FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# CRITICAL: Set CUDA architecture for Blackwell (RTX 5070)
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"
ENV CUDA_LAUNCH_BLOCKING=0
ENV TORCH_USE_CUDA_DSA=1

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# ⚡ SOLUTION 1: Install PyTorch 2.5.0 with CUDA 12.4 (better Blackwell support)
RUN pip3 install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Verify PyTorch installation
RUN python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"

# Copy and install requirements (WITHOUT PyTorch - already installed)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Verify no downgrades happened
RUN python3 -c "import torch; print(f'PyTorch after requirements: {torch.__version__}')"

# Copy application code
COPY main.py .
COPY model_manager.py .

# Create directories
RUN mkdir -p /app/models /app/outputs /app/cache /app/templates

# Copy templates
COPY templates/ /app/templates/

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
