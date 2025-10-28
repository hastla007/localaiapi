# 🔧 Troubleshooting Guide

Common issues and solutions for the AI API.

## 🚨 Installation Issues

### Docker Build Fails

**Error:** "failed to solve with frontend dockerfile.v0"

**Solutions:**
1. Update Docker Desktop to latest version
2. Clean Docker cache:
   ```bash
   docker system prune -a
   docker-compose build --no-cache
   ```

### GPU Not Detected

**Error:** `"gpu_available": false` in health check

**Solutions:**
1. Check Docker Desktop GPU settings:
   - Settings → Resources → WSL Integration
   - Ensure WSL 2 backend is enabled
   
2. Test GPU in Docker:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
   ```

3. Update NVIDIA drivers:
   - Visit https://www.nvidia.com/Download/index.aspx
   - Install latest drivers for your GPU

4. Restart Docker Desktop completely

### Port Already in Use

**Error:** "port is already allocated"

**Solution:** Change port in `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"  # Changed from 8000
```

Then use `http://localhost:8001` instead.

---

## 🐌 Performance Issues

### First Request Very Slow

**Symptom:** First image takes 5+ minutes

**Explanation:** This is normal! The first request must:
1. Download model from HuggingFace (~7-12 GB)
2. Load model into GPU memory
3. Compile CUDA kernels

**After first request, subsequent requests are 10-20x faster.**

**Solution:** Be patient on first run. Subsequent runs are fast.

### Out of Memory (OOM) Errors

**Error:** "CUDA out of memory"

**Solutions:**

1. **Reduce loaded models:**
   Edit `docker-compose.yml`:
   ```yaml
   environment:
     - MAX_LOADED_MODELS=1  # Changed from 2
   ```

2. **Use smaller models:**
   - Use BLIP-2 instead of LLaVA
   - Use SDXL instead of Flux
   
3. **Lower resolution:**
   ```json
   {
     "width": 512,   // Instead of 1024
     "height": 512,
     "num_inference_steps": 20  // Instead of 30
   }
   ```

4. **Unload models manually:**
   ```bash
   curl -X POST http://localhost:8000/api/unload-all
   ```

### Slow Generation After Working Fine

**Symptom:** Generation suddenly becomes slow

**Possible Causes:**
- Multiple models loaded (using more VRAM)
- System resources being used by other apps
- GPU thermal throttling

**Solutions:**
1. Unload unused models:
   ```bash
   curl http://localhost:8000/models  # Check loaded models
   curl -X POST http://localhost:8000/api/unload-all
   ```

2. Check GPU temperature:
   ```bash
   docker exec -it ai-api-local nvidia-smi
   ```

3. Restart container:
   ```bash
   docker-compose restart
   ```

---

## 🌐 Network Issues

### Cannot Connect to API

**Error:** "Connection refused" or "Cannot connect"

**Solutions:**

1. **Check if container is running:**
   ```bash
   docker-compose ps
   ```
   Should show `ai-api-local` as "Up"

2. **View logs:**
   ```bash
   docker-compose logs -f
   ```

3. **Restart container:**
   ```bash
   docker-compose restart
   ```

4. **Rebuild container:**
   ```bash
   docker-compose down
   docker-compose up -d --build
   ```

### Timeout Errors

**Error:** Request times out after 30s

**Solutions:**

1. **Increase timeout in n8n:**
   - HTTP Request node → Settings
   - Set timeout to 300000 ms (5 minutes)

2. **Use smaller parameters:**
   - Fewer inference steps
   - Lower resolution
   - Simpler models

---

## 📦 Model Download Issues

### Models Not Downloading

**Error:** "Failed to download model" or "Connection error"

**Solutions:**

1. **Check internet connection**

2. **Check HuggingFace status:**
   Visit: https://status.huggingface.co

3. **Try manual download:**
   ```bash
   docker exec -it ai-api-local bash
   huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0
   ```

4. **Use HuggingFace token for gated models:**
   - Get token: https://huggingface.co/settings/tokens
   - Add to `docker-compose.yml`:
     ```yaml
     environment:
       - HF_TOKEN=hf_your_token_here
     ```

### Disk Space Issues

**Error:** "No space left on device"

**Solutions:**

1. **Check available space:**
   ```bash
   docker exec -it ai-api-local df -h
   ```

2. **Clean Docker cache:**
   ```bash
   docker system prune -a
   ```

3. **Change cache location:**
   Edit `docker-compose.yml`:
   ```yaml
   volumes:
     - D:\ai-models:/app/cache  # Different drive
   ```

---

## 🔍 API Errors

### 500 Internal Server Error

**Solutions:**

1. **Check logs:**
   ```bash
   docker-compose logs -f ai-api
   ```

2. **Common causes:**
   - Invalid base64 image format
   - Missing required parameters
   - Model loading failed

3. **Restart with clean state:**
   ```bash
   curl -X POST http://localhost:8000/api/unload-all
   ```

### 404 Not Found

**Check:**
- URL is correct
- Endpoint exists: `http://localhost:8000/models`
- Port is correct (8000 by default)

### Invalid Base64 Image

**Error:** "Invalid base64 string" or "Cannot decode image"

**Solution:** Ensure base64 format:
```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAA..."
}
```

Or without prefix:
```json
{
  "image": "iVBORw0KGgoAAAA..."
}
```

---

## 🎯 n8n Integration Issues

### Workflow Times Out

**Solutions:**

1. **Increase timeout:**
   - HTTP Request node → Parameters
   - Set "Timeout" to 300000 (5 min)

2. **Split into separate workflows:**
   - Use webhook to trigger
   - Let first workflow complete
   - Chain to next workflow

### Base64 Image Not Working

**In n8n:**

1. **Convert file to base64:**
   ```javascript
   // In Code node
   const imageBuffer = $binary.data;
   const base64 = imageBuffer.toString('base64');
   return {
     image: `data:image/png;base64,${base64}`
   };
   ```

2. **Use from previous node:**
   ```
   {{ $json.image_base64 }}
   ```

---

## 🆘 Getting More Help

### Check Logs

Always check logs first:
```bash
docker-compose logs -f
```

### Check Container Status

```bash
docker-compose ps
docker stats ai-api-local
```

### Check GPU Status

```bash
docker exec -it ai-api-local nvidia-smi
```

### Interactive API Documentation

Visit: `http://localhost:8000/docs`

Test endpoints directly in browser.

### Full Reset

If all else fails:
```bash
# Stop everything
docker-compose down -v

# Clean Docker
docker system prune -a

# Delete cache (optional - will re-download models)
# rm -rf cache/
# rm -rf models/

# Rebuild from scratch
docker-compose build --no-cache
docker-compose up -d

# Watch logs
docker-compose logs -f
```

---

## 📊 Monitoring

### Check VRAM Usage

```bash
watch -n 1 docker exec ai-api-local nvidia-smi
```

### Check Model Status

```bash
curl http://localhost:8000/models | jq
```

### Monitor Docker Resources

```bash
docker stats ai-api-local
```

---

## 🎓 Common Misconceptions

**"Why is the first request so slow?"**
- Model download + loading is one-time cost
- Subsequent requests are much faster

**"The API is using too much VRAM"**
- This is intentional - keeps models loaded for speed
- Unload models manually if needed
- Adjust MAX_LOADED_MODELS for automatic management

**"Can I run this without GPU?"**
- Technically yes, but extremely slow (CPU inference)
- Not recommended - would take 10-30 minutes per image

---

**Still having issues? Check the logs - they're your best friend!**

```bash
docker-compose logs -f
```
