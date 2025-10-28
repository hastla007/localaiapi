# ⚡ Quick Start - Add Dashboard in 5 Minutes

## ✅ Pre-Installation Checklist

Make sure you have:
- [ ] AI API currently running (test at http://localhost:8000)
- [ ] Docker Desktop running
- [ ] Access to your `ai-api-project` folder
- [ ] Backup of current files (optional but recommended)

---

## 🚀 Installation Steps

### 1. Stop Your Container
```bash
cd path/to/ai-api-project
docker-compose down
```

### 2. Create Templates Directory
```bash
mkdir templates
```

### 3. Copy Dashboard HTML
Move `dashboard.html` into the `templates/` folder:
```
ai-api-project/
├── templates/
│   └── dashboard.html  ← Put it here
├── main.py
├── Dockerfile
└── ...
```

### 4. Replace These Files

**Option A: Manual Replacement**
1. Replace `main.py` with `main_updated.py` (rename it to `main.py`)
2. Replace `requirements.txt` with `requirements_updated.txt`
3. Replace `Dockerfile` with `Dockerfile_updated`

**Option B: Keep Originals (Safer)**
1. Rename current files:
   - `main.py` → `main_old.py`
   - `requirements.txt` → `requirements_old.txt`
   - `Dockerfile` → `Dockerfile_old`
2. Add new files with correct names

### 5. Update docker-compose.yml

Add this line to the `volumes` section:
```yaml
volumes:
  - ./models:/app/models
  - ./outputs:/app/outputs
  - ./cache:/app/cache
  - ./templates:/app/templates  # ← Add this line
```

Your `docker-compose.yml` should look like:
```yaml
version: '3.8'

services:
  ai-api:
    build: .
    container_name: ai-api-local
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./outputs:/app/outputs
      - ./cache:/app/cache
      - ./templates:/app/templates  # NEW LINE
    environment:
      - HF_HOME=/app/cache
      - TRANSFORMERS_CACHE=/app/cache
      - CUDA_VISIBLE_DEVICES=0
      - MAX_LOADED_MODELS=2
      - MODEL_TIMEOUT=300
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
```

### 6. Rebuild & Start
```bash
# Rebuild with new changes
docker-compose build

# Start the container
docker-compose up -d

# Watch logs to confirm startup
docker-compose logs -f
```

Look for: "Application startup complete"

### 7. Test the Dashboard
Open browser and go to:
```
http://localhost:8000/dashboard
```

You should see the dashboard! 🎉

---

## ✨ First Steps in Dashboard

1. **Overview Tab**: Check that your GPU is detected
2. **Click "Test Generation"**: Generate a test image
3. **Results Tab**: See your generated image
4. **Metrics Tab**: View generation time
5. **Logs Tab**: Check API activity

---

## 🔍 Verification Checklist

- [ ] Dashboard loads at http://localhost:8000/dashboard
- [ ] System status shows "online"
- [ ] GPU name is displayed correctly
- [ ] Test generation works
- [ ] Results appear in Results tab
- [ ] Logs show activity
- [ ] Original API endpoints still work: http://localhost:8000/

---

## 🆘 Troubleshooting

### Dashboard shows blank page
```bash
# Check logs
docker-compose logs -f

# Look for template errors
```

### "Templates directory not found"
```bash
# Make sure templates folder exists
ls -la templates/

# Check docker-compose.yml has the templates volume mounted
```

### Can't access dashboard
```bash
# Check container is running
docker-compose ps

# Should show "Up" status
```

### API works but dashboard doesn't
1. Make sure you copied `dashboard.html` to `templates/` folder
2. Verify `docker-compose.yml` has templates volume
3. Rebuild: `docker-compose down && docker-compose up -d --build`

---

## 🔄 Rolling Back

If something goes wrong:

```bash
# Stop container
docker-compose down

# Restore old files
mv main_old.py main.py
mv requirements_old.txt requirements.txt
mv Dockerfile_old Dockerfile

# Remove templates line from docker-compose.yml
# Rebuild
docker-compose build
docker-compose up -d
```

---

## 📚 Next Steps

Once dashboard is working:

1. Read `DASHBOARD_FEATURES.md` to learn all features
2. Explore each tab
3. Configure settings in Settings tab
4. Use dashboard while running n8n workflows
5. Monitor performance and VRAM usage

---

## 💡 Tips

- **Bookmark the dashboard**: http://localhost:8000/dashboard
- **Keep it open**: Monitor while running workflows
- **Check logs first**: When debugging issues
- **Unload models**: When switching between tasks

---

**Total time: ~5 minutes** ⚡

**Need help?** Check `DASHBOARD_INSTALLATION.md` for detailed guide.

---

✅ **You're all set! Enjoy your new dashboard!** 🎨
