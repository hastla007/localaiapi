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

# Install PyTorch with CUDA 12.4 support (for RTX 5070 Ti)
RUN pip3 install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install other requirements
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY model_manager.py .

# Create directories for models, outputs, and templates
RUN mkdir -p /app/models /app/outputs /app/cache /app/templates

# Copy templates directory if it exists
# Note: Make sure templates/ directory exists in your project root with dashboard.html inside
COPY templates/ /app/templates/

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
