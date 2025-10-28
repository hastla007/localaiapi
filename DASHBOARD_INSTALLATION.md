# 🎨 Dashboard Installation Guide

This guide will help you add the management dashboard to your existing AI API.

## 📋 What's New?

The dashboard adds:
- **Overview Page**: Real-time system status, GPU info, loaded models
- **Results Gallery**: View all generated images and videos
- **Metrics Page**: Charts showing generation times and API usage
- **Logs Page**: Real-time API logs for debugging
- **Settings Page**: Configure model management settings

## 🚀 Installation Steps

### Step 1: Backup Your Current Files

```bash
# Create a backup directory
mkdir -p backup
cp main.py backup/
cp requirements.txt backup/
cp Dockerfile backup/
```

### Step 2: Add Templates Directory

```bash
# Create templates directory
mkdir -p templates

# Copy the dashboard.html file to templates/
cp dashboard.html templates/
```

### Step 3: Replace Files

Replace these files with the updated versions:

1. **main.py** → Use `main_updated.py`
2. **requirements.txt** → Use `requirements_updated.txt`
3. **Dockerfile** → Use `Dockerfile_updated`

```bash
cp main_updated.py main.py
cp requirements_updated.txt requirements.txt
cp Dockerfile_updated Dockerfile
```

### Step 4: Rebuild and Restart

```bash
# Stop the current container
docker-compose down

# Rebuild with new changes
docker-compose build --no-cache

# Start the updated container
docker-compose up -d

# Watch logs to confirm it started
docker-compose logs -f
```

## 🎯 Accessing the Dashboard

Once the container is running:

1. Open your browser
2. Go to: **http://localhost:8000/dashboard**
3. You should see the new dashboard!

The API endpoints remain the same, so your n8n workflows won't be affected.

## ✨ Dashboard Features

### Overview Tab
- Real-time system status
- GPU information and VRAM usage
- Currently loaded models
- Quick actions (unload models, refresh status, test generation)

### Results Tab
- Gallery of all generated images
- Video outputs
- Download buttons for each result
- Automatic thumbnail generation

### Metrics Tab
- Generation time charts
- Requests per model
- Total statistics (requests, images, videos)
- Average generation times

### Logs Tab
- Real-time API logs
- Request tracking
- Error monitoring
- Clear logs function

### Settings Tab
- Configure max loaded models
- Set model timeout
- View current configuration
- Save settings (requires restart)

## 🔧 Troubleshooting

### Dashboard not loading?

**Check if API is running:**
```bash
docker-compose ps
```

**Check logs for errors:**
```bash
docker-compose logs -f
```

### Templates not found error?

Make sure the `templates` directory exists and contains `dashboard.html`:
```bash
ls -la templates/
```

If missing, copy the dashboard.html file again.

### Port conflict?

If port 8000 is already in use, change it in `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"  # Use 8001 instead
```

Then access dashboard at: http://localhost:8001/dashboard

## 📊 Using the Dashboard

### Quick Start

1. **Check System Status**: Open the Overview tab to see GPU and loaded models
2. **Generate a Test Image**: Click "Test Generation" button
3. **View Results**: Switch to Results tab to see your generated image
4. **Check Metrics**: View performance stats in Metrics tab
5. **Monitor Logs**: Use Logs tab to track API activity

### Best Practices

- **Monitor VRAM**: Keep an eye on the VRAM usage bar in Overview
- **Unload Models**: Use "Unload All Models" when switching between tasks
- **Check Logs**: Review logs if generation is slow or fails
- **Track Metrics**: Use metrics to optimize your generation parameters

## 🔄 Updating Later

If you need to update the dashboard in the future:

1. Update `templates/dashboard.html`
2. Restart the container:
   ```bash
   docker-compose restart
   ```

No rebuild needed for template changes!

## 🆘 Getting Help

If you encounter issues:

1. **Check API status**: http://localhost:8000/
2. **View container logs**: `docker-compose logs -f`
3. **Check GPU**: `docker exec -it ai-api-local nvidia-smi`
4. **Restart container**: `docker-compose restart`

## 📝 Notes

- The dashboard uses Tailwind CSS via CDN (no build step needed)
- Charts powered by Chart.js
- All data is stored in memory (resets on container restart)
- Settings are saved to `/app/settings.json` but require restart to apply

---

**Enjoy your new AI API Dashboard! 🎉**
