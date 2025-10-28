FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# ⚡ UPDATED: Install PyTorch 2.6.0 with CUDA 12.6 support (Blackwell compatible)
RUN pip3 install --no-cache-dir torch==2.6.0 torchvision==0.20.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# Install other requirements
RUN pip3 install --no-cache-dir -r requirements.txt

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
