from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import torch
import gc
import os
import time
import json
from datetime import datetime
from pathlib import Path
from collections import deque, defaultdict
import base64
from io import BytesIO
from PIL import Image

# Import model managers
from model_manager import ModelManager

# Initialize FastAPI
app = FastAPI(
    title="Multi-Model AI API",
    description="Local AI API for text-to-image, image-to-text, and video generation",
    version="1.0.0"
)

# Initialize templates
templates = Jinja2Templates(directory="/app/templates")

# Initialize model manager
model_manager = ModelManager()

# ==================== METRICS & LOGGING ====================

class MetricsTracker:
    def __init__(self, max_history=100):
        self.requests = deque(maxlen=max_history)
        self.generation_times = deque(maxlen=max_history)
        self.model_counts = defaultdict(int)
        self.logs = deque(maxlen=200)
        
    def log_request(self, model: str, generation_time: float, request_type: str):
        timestamp = datetime.now().isoformat()
        self.requests.append({
            "model": model,
            "time": generation_time,
            "type": request_type,
            "timestamp": timestamp
        })
        self.generation_times.append(generation_time)
        self.model_counts[model] += 1
        self.add_log(f"{request_type} request to {model} completed in {generation_time:.2f}s")
    
    def add_log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logs.append({
            "timestamp": timestamp,
            "message": message
        })
    
    def get_metrics(self):
        return {
            "total_requests": len(self.requests),
            "avg_generation_time": sum(self.generation_times) / len(self.generation_times) if self.generation_times else 0,
            "total_images": sum(1 for r in self.requests if r["type"] in ["text-to-image"]),
            "total_videos": sum(1 for r in self.requests if r["type"] == "video"),
            "requests_per_model": dict(self.model_counts),
            "recent_times": list(self.generation_times)[-20:]
        }
    
    def get_logs(self):
        return list(self.logs)
    
    def clear_logs(self):
        self.logs.clear()

metrics_tracker = MetricsTracker()
metrics_tracker.add_log("API started successfully")

# ==================== REQUEST MODELS ====================

class TextToImageRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for image generation")
    negative_prompt: Optional[str] = Field("", description="Negative prompt")
    width: Optional[int] = Field(1024, ge=512, le=2048)
    height: Optional[int] = Field(1024, ge=512, le=2048)
    num_inference_steps: Optional[int] = Field(30, ge=1, le=100)
    guidance_scale: Optional[float] = Field(7.5, ge=1.0, le=20.0)
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")

class ImageToTextRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")
    prompt: Optional[str] = Field("Describe this image in detail.", description="Optional prompt for captioning")
    max_length: Optional[int] = Field(200, ge=50, le=500)

class VideoGenerationRequest(BaseModel):
    image: Optional[str] = Field(None, description="Base64 encoded image for image-to-video")
    prompt: Optional[str] = Field(None, description="Text prompt for text-to-video")
    num_frames: Optional[int] = Field(14, ge=8, le=25)
    num_inference_steps: Optional[int] = Field(25, ge=10, le=50)
    fps: Optional[int] = Field(7, ge=5, le=30)
    motion_bucket_id: Optional[int] = Field(127, ge=1, le=255)
    noise_aug_strength: Optional[float] = Field(0.02, ge=0.0, le=1.0)

# ==================== HELPER FUNCTIONS ====================

def save_image(image: Image.Image, prefix: str = "generated") -> str:
    """Save image and return path"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.png"
    filepath = Path("/app/outputs") / filename
    image.save(filepath)
    return str(filepath)

def decode_base64_image(base64_str: str) -> Image.Image:
    """Decode base64 string to PIL Image"""
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data)).convert("RGB")

def encode_image_to_base64(image: Image.Image) -> str:
    """Encode PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# ==================== DASHBOARD ENDPOINTS ====================

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve the dashboard HTML page"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/api/dashboard/status")
async def dashboard_status():
    """Get system status for dashboard"""
    return {
        "status": "online",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "loaded_models": model_manager.get_loaded_models(),
        "model_stats": model_manager.get_model_stats(),
        "available_models": model_manager.AVAILABLE_MODELS
    }

@app.get("/api/dashboard/results")
async def dashboard_results():
    """Get list of generated results"""
    outputs_dir = Path("/app/outputs")
    results = []
    
    if outputs_dir.exists():
        for file in sorted(outputs_dir.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)[:50]:
            if file.suffix in ['.png', '.jpg', '.jpeg']:
                results.append({
                    "type": "image",
                    "path": f"/api/download/{file.name}",
                    "thumbnail": f"/api/download/{file.name}",
                    "filename": file.name,
                    "model": file.stem.split('_')[0],
                    "timestamp": datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                    "prompt": ""  # Could extract from metadata if stored
                })
            elif file.suffix in ['.mp4']:
                results.append({
                    "type": "video",
                    "path": f"/api/download/{file.name}",
                    "filename": file.name,
                    "model": "svd",
                    "timestamp": datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                })
    
    return {"results": results}

@app.get("/api/dashboard/metrics")
async def dashboard_metrics():
    """Get API metrics"""
    return metrics_tracker.get_metrics()

@app.get("/api/dashboard/logs")
async def dashboard_logs():
    """Get API logs"""
    return {"logs": metrics_tracker.get_logs()}

@app.post("/api/dashboard/logs/clear")
async def clear_logs():
    """Clear all logs"""
    metrics_tracker.clear_logs()
    metrics_tracker.add_log("Logs cleared by user")
    return {"success": True}

@app.get("/api/dashboard/settings")
async def get_settings():
    """Get current settings"""
    return {
        "max_loaded_models": model_manager.max_loaded_models,
        "model_timeout": model_manager.model_timeout,
        "cuda_device": os.getenv("CUDA_VISIBLE_DEVICES", "0"),
        "api_port": 8000
    }

@app.post("/api/dashboard/settings")
async def save_settings(settings: dict):
    """Save settings (note: requires restart for most settings)"""
    # For now, just save to a JSON file
    settings_file = Path("/app/settings.json")
    with open(settings_file, 'w') as f:
        json.dump(settings, f)
    return {"success": True, "message": "Settings saved. Restart container to apply changes."}

# ==================== HEALTH CHECK ====================

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Multi-Model AI API is running",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "loaded_models": model_manager.get_loaded_models(),
        "dashboard_url": "/dashboard"
    }

@app.get("/models")
def list_models():
    """List all available models and their status"""
    return {
        "available_models": model_manager.AVAILABLE_MODELS,
        "loaded_models": model_manager.get_loaded_models(),
        "model_stats": model_manager.get_model_stats()
    }

# ==================== TEXT-TO-IMAGE ENDPOINTS ====================

@app.post("/api/generate/flux")
async def generate_flux(request: TextToImageRequest):
    """Generate image using Flux.1-dev model"""
    try:
        metrics_tracker.add_log(f"Flux generation started: {request.prompt[:50]}...")
        start_time = time.time()
        
        # Load model
        pipe = model_manager.load_model("flux", "text-to-image")
        
        # Set seed if provided
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(request.seed)
        
        # Generate image
        image = pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            generator=generator
        ).images[0]
        
        # Save image
        filepath = save_image(image, "flux")
        
        generation_time = time.time() - start_time
        metrics_tracker.log_request("flux", generation_time, "text-to-image")
        
        return JSONResponse({
            "success": True,
            "model": "flux.1-dev",
            "image_path": filepath,
            "image_base64": encode_image_to_base64(image),
            "generation_time": round(generation_time, 2),
            "parameters": request.dict()
        })
        
    except Exception as e:
        metrics_tracker.add_log(f"ERROR in Flux generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate/sdxl")
async def generate_sdxl(request: TextToImageRequest):
    """Generate image using SDXL model"""
    try:
        metrics_tracker.add_log(f"SDXL generation started: {request.prompt[:50]}...")
        start_time = time.time()
        
        pipe = model_manager.load_model("sdxl", "text-to-image")
        
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(request.seed)
        
        image = pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            generator=generator
        ).images[0]
        
        filepath = save_image(image, "sdxl")
        generation_time = time.time() - start_time
        metrics_tracker.log_request("sdxl", generation_time, "text-to-image")
        
        return JSONResponse({
            "success": True,
            "model": "sdxl",
            "image_path": filepath,
            "image_base64": encode_image_to_base64(image),
            "generation_time": round(generation_time, 2),
            "parameters": request.dict()
        })
        
    except Exception as e:
        metrics_tracker.add_log(f"ERROR in SDXL generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate/sd3")
async def generate_sd3(request: TextToImageRequest):
    """Generate image using Stable Diffusion 3 model"""
    try:
        metrics_tracker.add_log(f"SD3 generation started: {request.prompt[:50]}...")
        start_time = time.time()
        
        pipe = model_manager.load_model("sd3", "text-to-image")
        
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(request.seed)
        
        image = pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            generator=generator
        ).images[0]
        
        filepath = save_image(image, "sd3")
        generation_time = time.time() - start_time
        metrics_tracker.log_request("sd3", generation_time, "text-to-image")
        
        return JSONResponse({
            "success": True,
            "model": "sd3",
            "image_path": filepath,
            "image_base64": encode_image_to_base64(image),
            "generation_time": round(generation_time, 2),
            "parameters": request.dict()
        })
        
    except Exception as e:
        metrics_tracker.add_log(f"ERROR in SD3 generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== IMAGE-TO-TEXT ENDPOINTS ====================

@app.post("/api/caption/llava")
async def caption_llava(request: ImageToTextRequest):
    """Generate caption using LLaVA model"""
    try:
        metrics_tracker.add_log("LLaVA caption started")
        start_time = time.time()
        
        # Decode image
        image = decode_base64_image(request.image)
        
        # Load model
        model, processor = model_manager.load_model("llava", "image-to-text")
        
        # Prepare inputs
        inputs = processor(
            images=image,
            text=request.prompt,
            return_tensors="pt"
        ).to("cuda")
        
        # Generate caption
        output = model.generate(
            **inputs,
            max_length=request.max_length,
            do_sample=True,
            temperature=0.7
        )
        
        caption = processor.decode(output[0], skip_special_tokens=True)
        
        generation_time = time.time() - start_time
        metrics_tracker.log_request("llava", generation_time, "image-to-text")
        
        return JSONResponse({
            "success": True,
            "model": "llava",
            "caption": caption,
            "generation_time": round(generation_time, 2)
        })
        
    except Exception as e:
        metrics_tracker.add_log(f"ERROR in LLaVA caption: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/caption/blip")
async def caption_blip(request: ImageToTextRequest):
    """Generate caption using BLIP-2 model"""
    try:
        metrics_tracker.add_log("BLIP-2 caption started")
        start_time = time.time()
        
        image = decode_base64_image(request.image)
        
        model, processor = model_manager.load_model("blip2", "image-to-text")
        
        inputs = processor(images=image, return_tensors="pt").to("cuda")
        
        output = model.generate(
            **inputs,
            max_length=request.max_length
        )
        
        caption = processor.decode(output[0], skip_special_tokens=True)
        
        generation_time = time.time() - start_time
        metrics_tracker.log_request("blip2", generation_time, "image-to-text")
        
        return JSONResponse({
            "success": True,
            "model": "blip2",
            "caption": caption,
            "generation_time": round(generation_time, 2)
        })
        
    except Exception as e:
        metrics_tracker.add_log(f"ERROR in BLIP-2 caption: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== VIDEO GENERATION ENDPOINTS ====================

@app.post("/api/video/svd")
async def generate_video_svd(request: VideoGenerationRequest):
    """Generate video using Stable Video Diffusion"""
    try:
        metrics_tracker.add_log("SVD video generation started")
        start_time = time.time()
        
        if not request.image:
            raise HTTPException(status_code=400, detail="Image is required for SVD")
        
        # Decode image
        image = decode_base64_image(request.image)
        image = image.resize((1024, 576))  # SVD recommended size
        
        # Load model
        pipe = model_manager.load_model("svd", "video-generation")
        
        # Generate video frames
        frames = pipe(
            image,
            num_frames=request.num_frames,
            num_inference_steps=request.num_inference_steps,
            motion_bucket_id=request.motion_bucket_id,
            noise_aug_strength=request.noise_aug_strength,
            decode_chunk_size=8
        ).frames[0]
        
        # Save video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = f"/app/outputs/svd_video_{timestamp}.mp4"
        
        # Export frames to video
        import cv2
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, request.fps, (width, height))
        
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        
        generation_time = time.time() - start_time
        metrics_tracker.log_request("svd", generation_time, "video")
        
        return JSONResponse({
            "success": True,
            "model": "stable-video-diffusion",
            "video_path": video_path,
            "num_frames": len(frames),
            "fps": request.fps,
            "generation_time": round(generation_time, 2)
        })
        
    except Exception as e:
        metrics_tracker.add_log(f"ERROR in SVD video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== UTILITY ENDPOINTS ====================

@app.post("/api/unload/{model_name}")
async def unload_model(model_name: str):
    """Manually unload a specific model"""
    try:
        model_manager.unload_model(model_name)
        metrics_tracker.add_log(f"Model {model_name} unloaded manually")
        return JSONResponse({
            "success": True,
            "message": f"Model {model_name} unloaded",
            "loaded_models": model_manager.get_loaded_models()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/unload-all")
async def unload_all_models():
    """Unload all models to free VRAM"""
    try:
        model_manager.unload_all()
        metrics_tracker.add_log("All models unloaded manually")
        return JSONResponse({
            "success": True,
            "message": "All models unloaded",
            "loaded_models": model_manager.get_loaded_models()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """Download generated file"""
    filepath = Path("/app/outputs") / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(filepath)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
