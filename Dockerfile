FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

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

# ⚡ STEP 1: Install PyTorch FIRST with Blackwell support
# Use nightly build for RTX 5070 (sm_120)
RUN pip3 install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu126

# Verify PyTorch installation
RUN python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# ⚡ STEP 2: Copy and install requirements (WITHOUT PyTorch pins)
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
