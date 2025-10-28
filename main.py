from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import torch
import gc
import os
import time
from datetime import datetime
from pathlib import Path
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

# Initialize model manager
model_manager = ModelManager()

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

# ==================== HEALTH CHECK ====================

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Multi-Model AI API is running",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "loaded_models": model_manager.get_loaded_models()
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
        
        return JSONResponse({
            "success": True,
            "model": "flux.1-dev",
            "image_path": filepath,
            "image_base64": encode_image_to_base64(image),
            "generation_time": round(generation_time, 2),
            "parameters": request.dict()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate/sdxl")
async def generate_sdxl(request: TextToImageRequest):
    """Generate image using SDXL model"""
    try:
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
        
        return JSONResponse({
            "success": True,
            "model": "sdxl",
            "image_path": filepath,
            "image_base64": encode_image_to_base64(image),
            "generation_time": round(generation_time, 2),
            "parameters": request.dict()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate/sd3")
async def generate_sd3(request: TextToImageRequest):
    """Generate image using Stable Diffusion 3 model"""
    try:
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
        
        return JSONResponse({
            "success": True,
            "model": "sd3",
            "image_path": filepath,
            "image_base64": encode_image_to_base64(image),
            "generation_time": round(generation_time, 2),
            "parameters": request.dict()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== IMAGE-TO-TEXT ENDPOINTS ====================

@app.post("/api/caption/llava")
async def caption_llava(request: ImageToTextRequest):
    """Generate caption using LLaVA model"""
    try:
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
        
        return JSONResponse({
            "success": True,
            "model": "llava",
            "caption": caption,
            "generation_time": round(generation_time, 2)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/caption/blip")
async def caption_blip(request: ImageToTextRequest):
    """Generate caption using BLIP-2 model"""
    try:
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
        
        return JSONResponse({
            "success": True,
            "model": "blip2",
            "caption": caption,
            "generation_time": round(generation_time, 2)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== VIDEO GENERATION ENDPOINTS ====================

@app.post("/api/video/svd")
async def generate_video_svd(request: VideoGenerationRequest):
    """Generate video using Stable Video Diffusion"""
    try:
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
        
        return JSONResponse({
            "success": True,
            "model": "stable-video-diffusion",
            "video_path": video_path,
            "num_frames": len(frames),
            "fps": request.fps,
            "generation_time": round(generation_time, 2)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== UTILITY ENDPOINTS ====================

@app.post("/api/unload/{model_name}")
async def unload_model(model_name: str):
    """Manually unload a specific model"""
    try:
        model_manager.unload_model(model_name)
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
