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
import numpy as np

# Import model managers
from model_manager import ModelManager
from comfyui_client import get_comfyui_client

# Initialize FastAPI
app = FastAPI(
    title="Multi-Model AI API",
    description="Local AI API with AnimateDiff Lightning & WAN 2.1 for ultra-fast video generation",
    version="1.2.0"
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
            "total_images": sum(1 for r in self.requests if r["type"] in ["text-to-image", "controlnet"]),
            "total_videos": sum(1 for r in self.requests if r["type"] == "video"),
            "requests_per_model": dict(self.model_counts),
            "recent_times": list(self.generation_times)[-20:]
        }
    
    def get_logs(self):
        return list(self.logs)
    
    def clear_logs(self):
        self.logs.clear()

metrics_tracker = MetricsTracker()
metrics_tracker.add_log("API started with AnimateDiff Lightning & WAN 2.1 (ultra-fast video generation)")

# ==================== REQUEST MODELS ====================

class TextToImageRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for image generation")
    negative_prompt: Optional[str] = Field("", description="Negative prompt")
    width: Optional[int] = Field(1024, ge=512, le=2048)
    height: Optional[int] = Field(1024, ge=512, le=2048)
    num_inference_steps: Optional[int] = Field(30, ge=1, le=100)
    guidance_scale: Optional[float] = Field(7.5, ge=1.0, le=20.0)
    seed: Optional[int] = Field(None)

class ControlNetRequest(BaseModel):
    prompt: str
    control_image: str
    negative_prompt: Optional[str] = Field("")
    controlnet_conditioning_scale: Optional[float] = Field(1.0, ge=0.1, le=2.0)
    width: Optional[int] = Field(1024, ge=512, le=2048)
    height: Optional[int] = Field(1024, ge=512, le=2048)
    num_inference_steps: Optional[int] = Field(30, ge=1, le=100)
    guidance_scale: Optional[float] = Field(7.5, ge=1.0, le=20.0)
    seed: Optional[int] = Field(None)

class ImageToTextRequest(BaseModel):
    image: str
    prompt: Optional[str] = Field("Describe this image in detail.")
    max_length: Optional[int] = Field(200, ge=50, le=500)

class VideoGenerationRequest(BaseModel):
    image: Optional[str] = Field(None)
    prompt: Optional[str] = Field(None)
    num_frames: Optional[int] = Field(14, ge=8, le=25)
    num_inference_steps: Optional[int] = Field(25, ge=10, le=50)
    fps: Optional[int] = Field(7, ge=5, le=30)
    motion_bucket_id: Optional[int] = Field(127, ge=1, le=255)
    noise_aug_strength: Optional[float] = Field(0.02, ge=0.0, le=1.0)

class AnimateDiffRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for video")
    negative_prompt: Optional[str] = Field("")
    num_frames: Optional[int] = Field(16, ge=8, le=32)
    num_inference_steps: Optional[int] = Field(8, ge=4, le=20, description="Lightning: use 4-8 steps")
    guidance_scale: Optional[float] = Field(1.5, ge=1.0, le=3.0, description="Lightning: use lower guidance")
    fps: Optional[int] = Field(8, ge=5, le=30)
    width: Optional[int] = Field(512, ge=256, le=768)
    height: Optional[int] = Field(512, ge=256, le=768)
    seed: Optional[int] = Field(None)

class WAN21Request(BaseModel):
    image: str = Field(..., description="Base64 encoded image for image-to-video")
    prompt: Optional[str] = Field("smooth camera movement, high quality", description="Optional guidance prompt")
    num_frames: Optional[int] = Field(49, ge=25, le=81, description="WAN 2.1 style: 49 frames ~2 sec")
    num_inference_steps: Optional[int] = Field(6, ge=4, le=12, description="LightX2V: 4-8 steps optimal")
    guidance_scale: Optional[float] = Field(1.5, ge=1.0, le=3.5, description="WAN 2.1: low CFG for speed")
    fps: Optional[int] = Field(24, ge=16, le=30)
    seed: Optional[int] = Field(None)

# ==================== HELPER FUNCTIONS ====================

def save_image(image: Image.Image, prefix: str = "generated") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.png"
    filepath = Path("/app/outputs") / filename
    image.save(filepath)
    return str(filepath)

def save_video(frames: List, prefix: str = "video", fps: int = 8) -> str:
    import cv2
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = f"/app/outputs/{prefix}_{timestamp}.mp4"
    
    if isinstance(frames[0], Image.Image):
        frames = [np.array(frame) for frame in frames]
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    for frame in frames:
        if frame.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
        out.write(frame_bgr)
    
    out.release()
    return video_path

def decode_base64_image(base64_str: str) -> Image.Image:
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data)).convert("RGB")

def encode_image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# ==================== DASHBOARD ENDPOINTS ====================

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/api/dashboard/status")
async def dashboard_status():
    return {
        "status": "online",
        "gpu_available": model_manager.cuda_compatible,
        "raw_cuda_available": torch.cuda.is_available(),
        "gpu_name": model_manager.device_name if torch.cuda.is_available() else "None",
        "cuda_capability": f"sm_{model_manager.device_capability[0]}{model_manager.device_capability[1]}" if model_manager.device_capability else None,
        "loaded_models": model_manager.get_loaded_models(),
        "model_stats": model_manager.get_model_stats(),
        "available_models": model_manager.AVAILABLE_MODELS
    }

@app.get("/api/dashboard/results")
async def dashboard_results():
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
                    "prompt": ""
                })
            elif file.suffix in ['.mp4']:
                results.append({
                    "type": "video",
                    "path": f"/api/download/{file.name}",
                    "filename": file.name,
                    "model": file.stem.split('_')[0] if '_' in file.stem else "video",
                    "timestamp": datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                })
    
    return {"results": results}

@app.get("/api/dashboard/metrics")
async def dashboard_metrics():
    return metrics_tracker.get_metrics()

@app.get("/api/dashboard/logs")
async def dashboard_logs():
    return {"logs": metrics_tracker.get_logs()}

@app.post("/api/dashboard/logs/clear")
async def clear_logs():
    metrics_tracker.clear_logs()
    metrics_tracker.add_log("Logs cleared by user")
    return {"success": True}

@app.get("/api/dashboard/settings")
async def get_settings():
    return {
        "max_loaded_models": model_manager.max_loaded_models,
        "model_timeout": model_manager.model_timeout,
        "cuda_device": os.getenv("CUDA_VISIBLE_DEVICES", "0"),
        "api_port": 8000
    }

@app.post("/api/dashboard/settings")
async def save_settings(settings: dict):
    settings_file = Path("/app/settings.json")
    with open(settings_file, 'w') as f:
        json.dump(settings, f)
    return {"success": True, "message": "Settings saved. Restart container to apply changes."}

# ==================== HEALTH CHECK ====================

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Multi-Model AI API v1.2 - AnimateDiff Lightning & WAN 2.1",
        "gpu_available": model_manager.cuda_compatible,
        "raw_cuda_available": torch.cuda.is_available(),
        "gpu_name": model_manager.device_name if torch.cuda.is_available() else "None",
        "cuda_capability": f"sm_{model_manager.device_capability[0]}{model_manager.device_capability[1]}" if model_manager.device_capability else None,
        "loaded_models": model_manager.get_loaded_models(),
        "total_models": len(model_manager.AVAILABLE_MODELS),
        "new_models": ["AnimateDiff Lightning (4-8 steps)", "WAN 2.1 + LightX2V (ultra-fast)"],
        "dashboard_url": "/dashboard"
    }

@app.get("/models")
def list_models():
    return {
        "available_models": model_manager.AVAILABLE_MODELS,
        "loaded_models": model_manager.get_loaded_models(),
        "model_stats": model_manager.get_model_stats()
    }

# ==================== TEXT-TO-IMAGE ENDPOINTS (keeping original code) ====================

@app.post("/api/generate/flux")
async def generate_flux(request: TextToImageRequest):
    try:
        metrics_tracker.add_log(f"Flux generation started: {request.prompt[:50]}...")
        start_time = time.time()
        pipe = model_manager.load_model("flux", "text-to-image")
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=model_manager.device.type).manual_seed(request.seed)
        image = pipe(
            prompt=request.prompt, negative_prompt=request.negative_prompt, width=request.width,
            height=request.height, num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale, generator=generator
        ).images[0]
        filepath = save_image(image, "flux")
        generation_time = time.time() - start_time
        metrics_tracker.log_request("flux", generation_time, "text-to-image")
        return JSONResponse({
            "success": True, "model": "flux.1-dev", "image_path": filepath,
            "image_base64": encode_image_to_base64(image), "generation_time": round(generation_time, 2),
            "parameters": request.dict()
        })
    except Exception as e:
        metrics_tracker.add_log(f"ERROR in Flux: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate/sdxl")
async def generate_sdxl(request: TextToImageRequest):
    try:
        metrics_tracker.add_log(f"SDXL generation started")
        start_time = time.time()
        pipe = model_manager.load_model("sdxl", "text-to-image")
        generator = None
        if request.seed:
            generator = torch.Generator(device=model_manager.device.type).manual_seed(request.seed)
        image = pipe(prompt=request.prompt, negative_prompt=request.negative_prompt, width=request.width,
                    height=request.height, num_inference_steps=request.num_inference_steps,
                    guidance_scale=request.guidance_scale, generator=generator).images[0]
        filepath = save_image(image, "sdxl")
        generation_time = time.time() - start_time
        metrics_tracker.log_request("sdxl", generation_time, "text-to-image")
        return JSONResponse({"success": True, "model": "sdxl", "image_path": filepath,
                           "image_base64": encode_image_to_base64(image), "generation_time": round(generation_time, 2)})
    except Exception as e:
        metrics_tracker.add_log(f"ERROR in SDXL: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate/sd3")
async def generate_sd3(request: TextToImageRequest):
    try:
        metrics_tracker.add_log("SD3 generation started")
        start_time = time.time()
        pipe = model_manager.load_model("sd3", "text-to-image")
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=model_manager.device.type).manual_seed(request.seed)
        image = pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            generator=generator,
        ).images[0]
        filepath = save_image(image, "sd3")
        generation_time = time.time() - start_time
        metrics_tracker.log_request("sd3", generation_time, "text-to-image")
        return JSONResponse({
            "success": True,
            "model": "stable-diffusion-3",
            "image_path": filepath,
            "image_base64": encode_image_to_base64(image),
            "generation_time": round(generation_time, 2),
            "parameters": request.dict(),
        })
    except Exception as e:
        metrics_tracker.add_log(f"ERROR in SD3: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate/pony")
async def generate_pony(request: TextToImageRequest):
    try:
        metrics_tracker.add_log("Pony generation started")
        start_time = time.time()
        pipe = model_manager.load_model("pony", "text-to-image")
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=model_manager.device.type).manual_seed(request.seed)
        image = pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            generator=generator,
        ).images[0]
        filepath = save_image(image, "pony")
        generation_time = time.time() - start_time
        metrics_tracker.log_request("pony", generation_time, "text-to-image")
        return JSONResponse({
            "success": True,
            "model": "pony-diffusion-v7",
            "image_path": filepath,
            "image_base64": encode_image_to_base64(image),
            "generation_time": round(generation_time, 2),
            "parameters": request.dict(),
        })
    except Exception as e:
        metrics_tracker.add_log(f"ERROR in Pony: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/caption/llava")
async def caption_llava(request: ImageToTextRequest):
    try:
        metrics_tracker.add_log("LLaVA captioning started")
        start_time = time.time()
        image = decode_base64_image(request.image)
        model, processor = model_manager.load_model("llava", "image-to-text")
        inputs = processor(
            images=image,
            text=request.prompt,
            return_tensors="pt"
        ).to(model.device)
        max_new_tokens = min(request.max_length, 256)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        if request.prompt:
            caption = caption.replace(request.prompt, "").strip()
        generation_time = time.time() - start_time
        metrics_tracker.log_request("llava", generation_time, "image-to-text")
        return JSONResponse({
            "success": True,
            "model": "llava-1.6",
            "caption": caption,
            "generation_time": round(generation_time, 2),
        })
    except Exception as e:
        metrics_tracker.add_log(f"ERROR in LLaVA: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/caption/blip")
async def caption_blip(request: ImageToTextRequest):
    try:
        metrics_tracker.add_log("BLIP captioning started")
        start_time = time.time()
        image = decode_base64_image(request.image)
        model, processor = model_manager.load_model("blip2", "image-to-text")
        inputs = processor(
            images=image,
            text=request.prompt,
            return_tensors="pt"
        ).to(model.device)
        max_new_tokens = min(request.max_length, 256)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        generation_time = time.time() - start_time
        metrics_tracker.log_request("blip2", generation_time, "image-to-text")
        return JSONResponse({
            "success": True,
            "model": "blip-2",
            "caption": caption,
            "generation_time": round(generation_time, 2),
        })
    except Exception as e:
        metrics_tracker.add_log(f"ERROR in BLIP: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/caption/qwen")
async def caption_qwen(request: ImageToTextRequest):
    try:
        metrics_tracker.add_log("Qwen captioning started")
        start_time = time.time()
        image = decode_base64_image(request.image)
        model, processor = model_manager.load_model("qwen", "image-to-text")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": request.prompt or "Describe this image in detail."}
            ],
        }]
        inputs = processor(messages, images=[image], return_tensors="pt").to(model.device)
        max_new_tokens = min(request.max_length, 256)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            generated_slice = generated_ids[:, input_ids.shape[-1]:]
        else:
            generated_slice = generated_ids
        generated_text = processor.batch_decode(
            generated_slice,
            skip_special_tokens=True,
        )[0].strip()
        generation_time = time.time() - start_time
        metrics_tracker.log_request("qwen", generation_time, "image-to-text")
        return JSONResponse({
            "success": True,
            "model": "qwen2-vl-2b",
            "caption": generated_text,
            "generation_time": round(generation_time, 2),
        })
    except Exception as e:
        metrics_tracker.add_log(f"ERROR in Qwen: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/controlnet/mistoline")
async def controlnet_mistoline(request: ControlNetRequest):
    try:
        metrics_tracker.add_log("MistoLine ControlNet generation started")
        start_time = time.time()
        pipe = model_manager.load_model("mistoline", "controlnet")
        control_image = decode_base64_image(request.control_image)
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=model_manager.device.type).manual_seed(request.seed)
        output = pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            image=control_image,
            controlnet_conditioning_scale=request.controlnet_conditioning_scale,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            generator=generator,
        )
        image = output.images[0]
        filepath = save_image(image, "mistoline")
        generation_time = time.time() - start_time
        metrics_tracker.log_request("mistoline", generation_time, "controlnet")
        return JSONResponse({
            "success": True,
            "model": "mistoline-controlnet",
            "image_path": filepath,
            "image_base64": encode_image_to_base64(image),
            "generation_time": round(generation_time, 2),
            "parameters": request.dict(exclude={"control_image"}),
        })
    except Exception as e:
        metrics_tracker.add_log(f"ERROR in MistoLine ControlNet: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/controlnet/union")
async def controlnet_union(request: ControlNetRequest):
    try:
        metrics_tracker.add_log("ControlNet Union generation started")
        start_time = time.time()
        pipe = model_manager.load_model("controlnet-union", "controlnet")
        control_image = decode_base64_image(request.control_image)
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=model_manager.device.type).manual_seed(request.seed)
        output = pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            image=control_image,
            controlnet_conditioning_scale=request.controlnet_conditioning_scale,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            generator=generator,
        )
        image = output.images[0]
        filepath = save_image(image, "controlnet-union")
        generation_time = time.time() - start_time
        metrics_tracker.log_request("controlnet-union", generation_time, "controlnet")
        return JSONResponse({
            "success": True,
            "model": "controlnet-union-sdxl",
            "image_path": filepath,
            "image_base64": encode_image_to_base64(image),
            "generation_time": round(generation_time, 2),
            "parameters": request.dict(exclude={"control_image"}),
        })
    except Exception as e:
        metrics_tracker.add_log(f"ERROR in ControlNet Union: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== VIDEO GENERATION ENDPOINTS ====================

@app.post("/api/video/svd")
async def generate_video_svd(request: VideoGenerationRequest):
    try:
        metrics_tracker.add_log("SVD video generation started")
        start_time = time.time()
        if not request.image:
            raise HTTPException(status_code=400, detail="Image required for SVD")
        image = decode_base64_image(request.image)
        image = image.resize((1024, 576))
        pipe = model_manager.load_model("svd", "video-generation")
        frames = pipe(image, num_frames=request.num_frames, num_inference_steps=request.num_inference_steps,
                     motion_bucket_id=request.motion_bucket_id, noise_aug_strength=request.noise_aug_strength,
                     decode_chunk_size=8).frames[0]
        video_path = save_video(frames, "svd", request.fps)
        generation_time = time.time() - start_time
        metrics_tracker.log_request("svd", generation_time, "video")
        return JSONResponse({"success": True, "model": "stable-video-diffusion", "video_path": video_path,
                           "num_frames": len(frames), "fps": request.fps, "generation_time": round(generation_time, 2)})
    except Exception as e:
        metrics_tracker.add_log(f"ERROR in SVD: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/video/animatediff")
async def generate_video_animatediff(request: AnimateDiffRequest):
    """AnimateDiff Lightning - Ultra-fast text-to-video (4-8 steps)"""
    try:
        metrics_tracker.add_log(f"AnimateDiff Lightning started: {request.prompt[:50]}...")
        start_time = time.time()
        
        pipe = model_manager.load_model("animatediff", "video-generation")
        
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=model_manager.device.type).manual_seed(request.seed)
        
        output = pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_frames=request.num_frames,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            width=request.width,
            height=request.height,
            generator=generator
        )
        
        frames = output.frames[0]
        video_path = save_video(frames, "animatediff", request.fps)
        
        generation_time = time.time() - start_time
        metrics_tracker.log_request("animatediff", generation_time, "video")
        
        return JSONResponse({
            "success": True,
            "model": "animatediff-lightning",
            "video_path": video_path,
            "num_frames": len(frames),
            "fps": request.fps,
            "generation_time": round(generation_time, 2),
            "note": "AnimateDiff Lightning uses 4-8 steps for ultra-fast generation",
            "parameters": request.dict()
        })
        
    except Exception as e:
        metrics_tracker.add_log(f"ERROR in AnimateDiff: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/video/wan21")
async def generate_video_wan21(request: WAN21Request):
    """WAN 2.1 + LightX2V - Ultra-fast image-to-video via ComfyUI"""
    try:
        metrics_tracker.add_log("WAN 2.1 video generation started (via ComfyUI)")
        start_time = time.time()

        # Decode the uploaded image
        image = decode_base64_image(request.image)

        # Acquire ComfyUI client and ensure service availability
        comfyui = get_comfyui_client()
        if not await comfyui.health_check():
            raise HTTPException(
                status_code=503,
                detail=(
                    "ComfyUI service is not available. Please ensure ComfyUI is running "
                    "with WAN 2.1 + LightX2V installed."
                )
            )

        video_path = await comfyui.generate_video_wan21(
            image=image,
            prompt=request.prompt,
            num_frames=request.num_frames,
            steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            fps=request.fps,
            seed=request.seed,
        )

        generation_time = time.time() - start_time
        metrics_tracker.log_request("wan21", generation_time, "video")

        return JSONResponse({
            "success": True,
            "model": "wan21-lightx2v",
            "video_path": video_path,
            "num_frames": request.num_frames,
            "fps": request.fps,
            "generation_time": round(generation_time, 2),
            "note": "Generated using native WAN 2.1 + LightX2V via ComfyUI",
            "parameters": request.dict(),
        })

    except HTTPException:
        raise
    except Exception as e:
        metrics_tracker.add_log(f"ERROR in WAN 2.1: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/comfyui/status")
async def comfyui_status():
    """Check ComfyUI service availability"""
    try:
        comfyui = get_comfyui_client()
        is_available = await comfyui.health_check()

        return {
            "available": is_available,
            "url": comfyui.base_url,
            "models_required": [
                "wan2.1_i2v.safetensors",
                "lightx2v_v0.1_lora.safetensors",
            ],
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e),
        }

# ==================== UTILITY ENDPOINTS ====================

@app.post("/api/unload/{model_name}")
async def unload_model(model_name: str):
    try:
        model_manager.unload_model(model_name)
        metrics_tracker.add_log(f"Model {model_name} unloaded")
        return JSONResponse({"success": True, "message": f"Model {model_name} unloaded",
                           "loaded_models": model_manager.get_loaded_models()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/unload-all")
async def unload_all_models():
    try:
        model_manager.unload_all()
        metrics_tracker.add_log("All models unloaded")
        return JSONResponse({"success": True, "message": "All models unloaded",
                           "loaded_models": model_manager.get_loaded_models()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    filepath = Path("/app/outputs") / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(filepath)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
