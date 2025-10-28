import torch
import gc
import time
from typing import Dict, Any, Optional, Tuple
import os

# Import diffusers components
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusion3Pipeline,
    StableVideoDiffusionPipeline,
    StableDiffusionPipeline,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    DPMSolverMultistepScheduler
)

# Try to import FluxPipeline (requires diffusers >= 0.30.0)
try:
    from diffusers import FluxPipeline
    FLUX_AVAILABLE = True
except ImportError:
    FLUX_AVAILABLE = False
    print("Warning: FluxPipeline not available. Install diffusers>=0.30.0 for Flux support.")

from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    Blip2Processor,
    Blip2ForConditionalGeneration,
    Qwen2VLForConditionalGeneration
)


class ModelManager:
    """Manages multiple AI models with lazy loading and automatic unloading"""
    
    AVAILABLE_MODELS = {
        "flux": {
            "name": "Flux.1-dev",
            "type": "text-to-image",
            "model_id": "black-forest-labs/FLUX.1-dev",
            "vram_gb": 12,
            "category": "General"
        },
        "sdxl": {
            "name": "Stable Diffusion XL",
            "type": "text-to-image",
            "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
            "vram_gb": 7,
            "category": "General"
        },
        "sd3": {
            "name": "Stable Diffusion 3",
            "type": "text-to-image",
            "model_id": "stabilityai/stable-diffusion-3-medium-diffusers",
            "vram_gb": 10,
            "category": "General"
        },
        "pony": {
            "name": "Pony Diffusion V7",
            "type": "text-to-image",
            "model_id": "purplesmartai/pony-v7-base",
            "vram_gb": 5,
            "category": "Anime/Cartoon"
        },
        "llava": {
            "name": "LLaVA 1.6",
            "type": "image-to-text",
            "model_id": "llava-hf/llava-1.5-7b-hf",
            "vram_gb": 13,
            "category": "Vision"
        },
        "blip2": {
            "name": "BLIP-2",
            "type": "image-to-text",
            "model_id": "Salesforce/blip2-opt-2.7b",
            "vram_gb": 6,
            "category": "Vision"
        },
        "qwen": {
            "name": "Qwen-Image",
            "type": "image-to-text",
            "model_id": "Qwen/Qwen2-VL-2B-Instruct",
            "vram_gb": 8,
            "category": "Vision"
        },
        "svd": {
            "name": "Stable Video Diffusion",
            "type": "video-generation",
            "model_id": "stabilityai/stable-video-diffusion-img2vid-xt",
            "vram_gb": 8,
            "category": "Video"
        },
        "mistoline": {
            "name": "MistoLine",
            "type": "controlnet",
            "model_id": "TheMistoAI/MistoLine",
            "base_model": "sdxl",
            "vram_gb": 8,
            "category": "ControlNet"
        },
        "controlnet-union": {
            "name": "ControlNet Union SDXL",
            "type": "controlnet",
            "model_id": "xinsir/controlnet-union-sdxl-1.0",
            "base_model": "sdxl",
            "vram_gb": 9,
            "category": "ControlNet"
        }
    }
    
    def __init__(self, max_loaded_models: int = 2, model_timeout: int = 300):
        """
        Initialize model manager
        
        Args:
            max_loaded_models: Maximum number of models to keep loaded
            model_timeout: Time in seconds before unloading unused model
        """
        self.loaded_models: Dict[str, Any] = {}
        self.model_last_used: Dict[str, float] = {}
        self.max_loaded_models = int(os.getenv("MAX_LOADED_MODELS", max_loaded_models))
        self.model_timeout = int(os.getenv("MODEL_TIMEOUT", model_timeout))
        
        print(f"Model Manager initialized:")
        print(f"  Max loaded models: {self.max_loaded_models}")
        print(f"  Model timeout: {self.model_timeout}s")
        print(f"  GPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    def _cleanup_old_models(self):
        """Unload old models if we're at capacity"""
        if len(self.loaded_models) >= self.max_loaded_models:
            # Sort by last used time
            sorted_models = sorted(
                self.model_last_used.items(),
                key=lambda x: x[1]
            )
            
            # Unload oldest model
            oldest_model = sorted_models[0][0]
            print(f"Unloading {oldest_model} to make room for new model")
            self.unload_model(oldest_model)
    
    def _load_text_to_image_model(self, model_key: str):
        """Load a text-to-image diffusion model"""
        model_info = self.AVAILABLE_MODELS[model_key]
        model_id = model_info["model_id"]
        
        print(f"Loading {model_info['name']} from {model_id}...")
        
        if model_key == "flux":
            if not FLUX_AVAILABLE:
                raise ImportError(
                    "FluxPipeline is not available. "
                    "Please upgrade diffusers: pip install diffusers>=0.30.0"
                )
            pipe = FluxPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                use_safetensors=True
            )
        elif model_key == "sdxl":
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        elif model_key == "sd3":
            pipe = StableDiffusion3Pipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                use_safetensors=True
            )
        elif model_key == "pony":
            # Pony is SD 1.5 based
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                safety_checker=None
            )
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        else:
            raise ValueError(f"Unknown text-to-image model: {model_key}")
        
        # Enable memory optimizations
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
        
        # Try to enable xformers if available
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("  xformers enabled for memory efficiency")
        except:
            print("  xformers not available, using default attention")
        
        pipe = pipe.to("cuda")
        print(f"  {model_info['name']} loaded successfully")
        
        return pipe
    
    def _load_image_to_text_model(self, model_key: str) -> Tuple[Any, Any]:
        """Load an image-to-text model"""
        model_info = self.AVAILABLE_MODELS[model_key]
        model_id = model_info["model_id"]
        
        print(f"Loading {model_info['name']} from {model_id}...")
        
        if model_key == "llava":
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        elif model_key == "blip2":
            processor = Blip2Processor.from_pretrained(model_id)
            model = Blip2ForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        elif model_key == "qwen":
            processor = AutoProcessor.from_pretrained(model_id)
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            raise ValueError(f"Unknown image-to-text model: {model_key}")
        
        print(f"  {model_info['name']} loaded successfully")
        
        return model, processor
    
    def _load_video_generation_model(self, model_key: str):
        """Load a video generation model"""
        model_info = self.AVAILABLE_MODELS[model_key]
        model_id = model_info["model_id"]
        
        print(f"Loading {model_info['name']} from {model_id}...")
        
        if model_key == "svd":
            pipe = StableVideoDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                variant="fp16"
            )
            pipe.enable_model_cpu_offload()
        else:
            raise ValueError(f"Unknown video generation model: {model_key}")
        
        print(f"  {model_info['name']} loaded successfully")
        
        return pipe
    
    def _load_controlnet_model(self, model_key: str):
        """Load a ControlNet model with SDXL pipeline"""
        model_info = self.AVAILABLE_MODELS[model_key]
        model_id = model_info["model_id"]
        
        print(f"Loading {model_info['name']} from {model_id}...")
        
        # Load ControlNet
        controlnet = ControlNetModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        )
        
        # Load base SDXL pipeline with ControlNet
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_model_id,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            variant="fp16"
        )
        
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
        
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("  xformers enabled for ControlNet")
        except:
            print("  xformers not available")
        
        pipe = pipe.to("cuda")
        print(f"  {model_info['name']} loaded successfully")
        
        return pipe
    
    def load_model(self, model_key: str, model_type: str):
        """
        Load a model with lazy loading
        
        Args:
            model_key: Key of the model to load
            model_type: Type of model (text-to-image, image-to-text, video-generation, controlnet)
            
        Returns:
            Loaded model or (model, processor) tuple
        """
        if model_key not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_key}")
        
        # Check if model is already loaded
        if model_key in self.loaded_models:
            print(f"Using cached {model_key} model")
            self.model_last_used[model_key] = time.time()
            return self.loaded_models[model_key]
        
        # Clean up old models if needed
        self._cleanup_old_models()
        
        # Load the model based on type
        if model_type == "text-to-image":
            model = self._load_text_to_image_model(model_key)
        elif model_type == "image-to-text":
            model = self._load_image_to_text_model(model_key)
        elif model_type == "video-generation":
            model = self._load_video_generation_model(model_key)
        elif model_type == "controlnet":
            model = self._load_controlnet_model(model_key)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Cache the model
        self.loaded_models[model_key] = model
        self.model_last_used[model_key] = time.time()
        
        # Print VRAM usage
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1e9
            vram_cached = torch.cuda.memory_reserved() / 1e9
            print(f"  VRAM used: {vram_used:.2f} GB, cached: {vram_cached:.2f} GB")
        
        return model
    
    def unload_model(self, model_key: str):
        """Unload a specific model"""
        if model_key in self.loaded_models:
            print(f"Unloading {model_key}...")
            del self.loaded_models[model_key]
            del self.model_last_used[model_key]
            
            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache()
            
            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated() / 1e9
                print(f"  VRAM after unload: {vram_used:.2f} GB")
    
    def unload_all(self):
        """Unload all models"""
        print("Unloading all models...")
        self.loaded_models.clear()
        self.model_last_used.clear()
        
        gc.collect()
        torch.cuda.empty_cache()
        
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1e9
            print(f"  VRAM after clearing all: {vram_used:.2f} GB")
    
    def get_loaded_models(self) -> list:
        """Get list of currently loaded models"""
        return list(self.loaded_models.keys())
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about models"""
        stats = {}
        for model_key, last_used in self.model_last_used.items():
            time_since_used = time.time() - last_used
            stats[model_key] = {
                "loaded": True,
                "last_used_seconds_ago": round(time_since_used, 2),
                "vram_estimate_gb": self.AVAILABLE_MODELS[model_key]["vram_gb"]
            }
        return stats
