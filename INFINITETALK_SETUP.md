# InfiniteTalk Integration Setup

## Installation Steps

### 1. Stop Container
```bash
docker-compose down
```

### 2. Update Files
Replace the following files with the updated versions:
- `model_manager.py`
- `main.py`
- `requirements.txt`
- `templates/dashboard.html`

### 3. Rebuild Container
```bash
docker-compose build --no-cache
docker-compose up -d
```

### 4. Test Installation
```bash
# Check if InfiniteTalk is listed
curl http://localhost:8000/models
```

## Usage

### Via API

**Endpoint:** `POST /api/talking-head/infinitetalk`

**With Audio:**
```json
{
  "face_image": "data:image/png;base64,...",
  "audio": "data:audio/wav;base64,...",
  "num_frames": 120,
  "fps": 25,
  "expression_scale": 1.0,
  "head_motion_scale": 1.0
}
```

**With Text:**
```json
{
  "face_image": "data:image/png;base64,...",
  "text": "Hello, this is a test of the InfiniteTalk system.",
  "num_frames": 120,
  "fps": 25,
  "expression_scale": 1.2,
  "head_motion_scale": 0.8
}
```
