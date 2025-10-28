# Multi-Model AI API for n8n Automation

Local AI API with text-to-image, image-to-text, and video generation capabilities. Optimized for Docker + NVIDIA GPU with intelligent model management.

## 🚀 Features

- **Multiple AI Models** with lazy loading and automatic VRAM management
- **Text-to-Image**: Flux.1-dev, SDXL, Stable Diffusion 3
- **Image-to-Text**: LLaVA 1.6, BLIP-2
- **Video Generation**: Stable Video Diffusion (SVD)
- **RESTful API** perfect for n8n automations
- **GPU Optimized** with automatic model unloading
- **Base64 & File Support** for easy integration

## 📋 Requirements

- Windows 11 with Docker Desktop
- NVIDIA GPU with Docker support (already configured ✅)
- ~50GB free disk space for models
- 16GB+ VRAM recommended

## 🛠️ Installation

### Step 1: Extract and Navigate

```bash
# Extract the project folder
# Open PowerShell or Command Prompt
cd path\to\ai-api-project
```

### Step 2: Build Docker Image

```bash
docker-compose build
```

This will take 10-20 minutes on first build.

### Step 3: Start the API

```bash
docker-compose up -d
```

### Step 4: Verify It's Running

Open browser: `http://localhost:8000`

You should see:
```json
{
  "status": "online",
  "message": "Multi-Model AI API is running",
  "gpu_available": true,
  "gpu_name": "NVIDIA Blackwell...",
  "loaded_models": []
}
```

## 🎯 API Endpoints

### Health Check
```bash
GET http://localhost:8000/
GET http://localhost:8000/models
```

### Text-to-Image

**Flux.1-dev** (Best Quality)
```bash
POST http://localhost:8000/api/generate/flux
```

**SDXL** (Faster)
```bash
POST http://localhost:8000/api/generate/sdxl
```

**Stable Diffusion 3**
```bash
POST http://localhost:8000/api/generate/sd3
```

### Image-to-Text

**LLaVA** (Detailed Captions)
```bash
POST http://localhost:8000/api/caption/llava
```

**BLIP-2** (Fast Captions)
```bash
POST http://localhost:8000/api/caption/blip
```

### Video Generation

**Stable Video Diffusion**
```bash
POST http://localhost:8000/api/video/svd
```

### Utility

```bash
POST http://localhost:8000/api/unload/{model_name}
POST http://localhost:8000/api/unload-all
GET  http://localhost:8000/api/download/{filename}
```

## 📝 n8n Integration Examples

### Example 1: Generate Image with Flux

In n8n, add an **HTTP Request** node:

**Method**: POST  
**URL**: `http://localhost:8000/api/generate/flux`  
**Body Content Type**: JSON  
**Body**:
```json
{
  "prompt": "A majestic dragon flying over mountains at sunset, digital art",
  "negative_prompt": "blurry, low quality, distorted",
  "width": 1024,
  "height": 1024,
  "num_inference_steps": 30,
  "guidance_scale": 7.5,
  "seed": 42
}
```

**Response** includes:
- `image_base64`: Base64 encoded image (use directly in n8n)
- `image_path`: File path in container
- `generation_time`: Time taken

### Example 2: Caption an Image with LLaVA

**Method**: POST  
**URL**: `http://localhost:8000/api/caption/llava`  
**Body**:
```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAA...",
  "prompt": "Describe this image in detail",
  "max_length": 200
}
```

In n8n, you can convert image files to base64 using the **Convert to File** or **Code** node.

### Example 3: Generate Video from Image

**Method**: POST  
**URL**: `http://localhost:8000/api/video/svd`  
**Body**:
```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAA...",
  "num_frames": 14,
  "num_inference_steps": 25,
  "fps": 7,
  "motion_bucket_id": 127
}
```

### Example 4: n8n Workflow Pattern

```
1. [Webhook Trigger] 
   ↓ (receives text prompt)
2. [HTTP Request: Generate Image]
   ↓ (returns base64 image)
3. [HTTP Request: Caption Image]
   ↓ (describes the image)
4. [HTTP Request: Generate Video]
   ↓ (creates video from image)
5. [Send to Email/Slack/etc]
```

## 🔧 Configuration

Edit `docker-compose.yml` to customize:

```yaml
environment:
  - MAX_LOADED_MODELS=2        # How many models to keep in VRAM
  - MODEL_TIMEOUT=300          # Seconds before auto-unload
  - CUDA_VISIBLE_DEVICES=0     # GPU to use
```

**Recommended Settings by VRAM:**
- **20GB VRAM**: `MAX_LOADED_MODELS=2`
- **40GB VRAM**: `MAX_LOADED_MODELS=3`
- **80GB VRAM**: `MAX_LOADED_MODELS=4` or more

## 📦 Model Downloads

Models download automatically on first use. Download sizes:

- **Flux.1-dev**: ~12GB
- **SDXL**: ~7GB
- **SD3**: ~10GB
- **LLaVA**: ~13GB
- **BLIP-2**: ~6GB
- **SVD**: ~8GB

Models are cached in `./cache` folder and persist between restarts.

## 🐛 Troubleshooting

### GPU Not Detected

```bash
# Check if Docker can see GPU
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

### Port Already in Use

Change port in `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"  # Use 8001 instead
```

### Out of Memory

Reduce `MAX_LOADED_MODELS` or use smaller models (BLIP-2, SDXL).

### Check Logs

```bash
docker-compose logs -f
```

### Restart Container

```bash
docker-compose restart
```

## 🎨 Full cURL Examples

### Generate Image
```bash
curl -X POST http://localhost:8000/api/generate/flux \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "cyberpunk city at night, neon lights",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 30,
    "guidance_scale": 7.5
  }'
```

### Caption Image
```bash
curl -X POST http://localhost:8000/api/caption/llava \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/png;base64,YOUR_BASE64_HERE",
    "prompt": "What is in this image?"
  }'
```

### Check Status
```bash
curl http://localhost:8000/models
```

### Unload All Models (Free VRAM)
```bash
curl -X POST http://localhost:8000/api/unload-all
```

## 📊 API Response Format

All endpoints return JSON:

**Success Response:**
```json
{
  "success": true,
  "model": "flux.1-dev",
  "image_base64": "iVBORw0KGgo...",
  "image_path": "/app/outputs/flux_20241028_143022.png",
  "generation_time": 12.34,
  "parameters": {...}
}
```

**Error Response:**
```json
{
  "detail": "Error message here"
}
```

## 📂 File Structure

```
ai-api-project/
├── docker-compose.yml    # Docker orchestration
├── Dockerfile           # Container setup
├── requirements.txt     # Python dependencies
├── main.py             # FastAPI application
├── model_manager.py    # Model loading logic
├── models/             # (auto-created) Model cache
├── outputs/            # (auto-created) Generated files
└── cache/              # (auto-created) HuggingFace cache
```

## 🔒 Security Notes

- This API has **no authentication** - it's meant for local use
- Don't expose port 8000 to the internet
- For production, add API keys/tokens

## 🚦 Starting & Stopping

**Start:**
```bash
docker-compose up -d
```

**Stop:**
```bash
docker-compose down
```

**View Logs:**
```bash
docker-compose logs -f
```

**Rebuild After Changes:**
```bash
docker-compose up -d --build
```

## 🎓 Tips for n8n

1. **Use Variables**: Store `http://localhost:8000` as a credential
2. **Error Handling**: Add error outputs to handle timeouts
3. **Webhooks**: Trigger generations via webhook for async workflows
4. **File Storage**: Save `image_base64` to Google Drive, Dropbox, etc.
5. **Chaining**: Use one endpoint's output as another's input

## 💡 Performance Tips

- First request per model is slow (loading time)
- Subsequent requests are fast (model cached)
- Use smaller `num_inference_steps` for faster results (20-25)
- Lower resolution = faster generation
- SVD video generation is resource-intensive (~1-2 min)

## 📈 Monitoring VRAM

Check inside container:
```bash
docker exec -it ai-api-local nvidia-smi
```

Or call:
```bash
curl http://localhost:8000/models
```

## 🆘 Support

- **Models not downloading?** Check internet connection and HuggingFace status
- **Slow generation?** Normal on first load, speeds up after
- **API not responding?** Check `docker-compose logs -f`

## 📜 License

Free to use for personal and commercial projects.

---

**Built for n8n automation workflows with ❤️**
