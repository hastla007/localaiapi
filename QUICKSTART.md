# 🚀 Quick Start Guide

Get your AI API running in 5 minutes!

## Step-by-Step Setup

### 1️⃣ Extract Project
Extract the `ai-api-project` folder to your preferred location, e.g.:
```
C:\Users\YourName\ai-api-project
```

### 2️⃣ Open Terminal
Open PowerShell or Command Prompt and navigate to the project:
```bash
cd C:\Users\YourName\ai-api-project
```

### 3️⃣ Build Container (First Time Only)
```bash
docker-compose build
```
⏱️ This takes 10-20 minutes - perfect time for coffee! ☕

### 4️⃣ Start the API
```bash
docker-compose up -d
```

### 5️⃣ Verify It's Running
Open your browser and go to:
```
http://localhost:8000
```

You should see:
```json
{
  "status": "online",
  "gpu_available": true,
  ...
}
```

🎉 **Congratulations! Your API is running!**

---

## Testing Your Setup

### Option A: Use Web Browser
Visit: `http://localhost:8000/docs`

This opens the interactive API documentation where you can test all endpoints.

### Option B: Run Test Script
```bash
# Install Python requests library if needed
pip install requests pillow

# Run the test script
python test_api.py
```

### Option C: Use cURL
```bash
# Check status
curl http://localhost:8000/

# Generate an image
curl -X POST http://localhost:8000/api/generate/sdxl \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"a cat\", \"width\": 512, \"height\": 512, \"num_inference_steps\": 20}"
```

---

## First Image Generation

Your first image will take 2-3 minutes because:
1. ✅ Model downloads from HuggingFace (~7GB for SDXL)
2. ✅ Model loads into GPU memory
3. ✅ Image generates

**After that, subsequent images are MUCH faster (15-30 seconds)!**

---

## n8n Integration - Quick Test

### 1. Create HTTP Request Node in n8n

**Method:** POST  
**URL:** `http://localhost:8000/api/generate/sdxl`  
**Body (JSON):**
```json
{
  "prompt": "a beautiful landscape",
  "width": 512,
  "height": 512,
  "num_inference_steps": 20
}
```

### 2. Execute the Node

You'll get back:
- `image_base64` - Use this in other n8n nodes
- `image_path` - File location in container
- `generation_time` - How long it took

### 3. Use the Image

The `image_base64` can be:
- Sent via email
- Posted to Slack/Discord
- Uploaded to Google Drive
- Used in another AI model
- Anything else!

---

## Common First-Time Issues

### ❌ "Cannot connect to API"
**Solution:** Make sure Docker is running
```bash
docker-compose ps
```

### ❌ "GPU not available"
**Solution:** Check Docker Desktop settings
- Settings → Resources → WSL Integration
- Make sure WSL 2 backend is enabled
- Restart Docker Desktop

### ❌ Port 8000 already in use
**Solution:** Change port in `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"  # Use 8001 instead
```

### ❌ Out of disk space
**Solution:** Models need ~50GB. Free up space or change cache location in `docker-compose.yml`

---

## Useful Commands

### View Logs
```bash
docker-compose logs -f
```

### Restart API
```bash
docker-compose restart
```

### Stop API
```bash
docker-compose down
```

### Check GPU Usage
```bash
docker exec -it ai-api-local nvidia-smi
```

### Remove Everything (Fresh Start)
```bash
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

---

## Next Steps

1. ✅ Read `README.md` for full documentation
2. ✅ Import `n8n-workflow-example.json` into n8n (see file)
3. ✅ Explore `/docs` endpoint for interactive API testing
4. ✅ Try different models (Flux for best quality, SDXL for speed)
5. ✅ Build your automation workflows!

---

## Support & Tips

- **Slow first request?** Normal! Models are downloading/loading
- **Want faster results?** Use fewer `num_inference_steps` (20-25)
- **Need better quality?** Use Flux.1 instead of SDXL
- **VRAM errors?** Reduce `MAX_LOADED_MODELS` in docker-compose.yml

---

**Ready to create some AI magic! 🎨✨**
