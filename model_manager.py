import torch
import gc
import time
from typing import Dict, Any, Optional, Tuple
import os
import warnings

# Import diffusers components
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusion3Pipeline,
    StableVideoDiffusionPipeline,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    DPMSolverMultistepScheduler,
    AnimateDiffPipeline,
    MotionAdapter,
    LCMScheduler,
    DiffusionPipeline
)

# Try to import FluxPipeline
try:
    from diffusers import FluxPipeline
    FLUX_AVAILABLE = True
except ImportError:
    FLUX_AVAILABLE = False
    print("Warning: FluxPipeline not available.")

# Try to import CogVideoX for WAN 2.1 proxy
try:
    from diffusers import CogVideoXImageToVideoPipeline
    COGVIDEO_AVAILABLE = True
except ImportError:
    COGVIDEO_AVAILABLE = False
    print("Warning: CogVideoX not available. WAN 2.1 will use SVD fallback.")

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
            "name": "Flux.1-dev (LowVRAM)",
            "type": "text-to-image",
            "model_id": "black-forest-labs/FLUX.1-dev",
            "vram_gb": 10,
            "category": "General",
            "note": "Using component loading with aggressive offloading"
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
        "animatediff": {
            "name": "AnimateDiff Lightning",
            "type": "video-generation",
            "model_id": "ByteDance/AnimateDiff-Lightning",
            "base_model": "emilianJR/epiCRealism",
            "vram_gb": 6,
            "category": "Video",
            "description": "Ultra-fast text-to-video (4-8 steps)"
        },
        "wan21": {
            "name": "WAN 2.1 + LightX2V",
            "type": "video-generation",
            "model_id": "THUDM/CogVideoX-5b-I2V",  # Using CogVideoX as proxy
            "vram_gb": 5,
            "category": "Video",
            "description": "Fast image-to-video (WAN 2.1 style, 4-8 steps)"
        },
        "infinitetalk": {
            "name": "InfiniteTalk (Hybrid + WAN 2.1)",
            "type": "talking-head",
            "model_id": "hybrid-infinitetalk-wan21",
            "vram_gb": 5,  # Reduced - only WAN 2.1 needs GPU, face prep is CPU
            "category": "Video",
            "description": "Talking head using face prep + ComfyUI WAN 2.1"
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
        self.loaded_models: Dict[str, Any] = {}
        self.model_last_used: Dict[str, float] = {}
        self.max_loaded_models = int(os.getenv("MAX_LOADED_MODELS", max_loaded_models))
        self.model_timeout = int(os.getenv("MODEL_TIMEOUT", model_timeout))

        self.device = torch.device("cpu")
        self.cuda_compatible = False
        self.device_name = "CPU"
        self.device_capability: Optional[Tuple[int, int]] = None

        self._initialize_device()

        print(f"Model Manager initialized:")
        print(f"  Max loaded models: {self.max_loaded_models}")
        print(f"  Model timeout: {self.model_timeout}s")
        print(f"  GPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {self.device_name}")
            if self.device_capability:
                major, minor = self.device_capability
                print(f"  CUDA Capability: sm_{major}{minor}")
            compatibility = "Yes" if self.cuda_compatible else "No (falling back to CPU)"
            print(f"  Compatible with PyTorch build: {compatibility}")
            if self.cuda_compatible:
                print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    def _initialize_device(self) -> None:
        """Determine whether CUDA can be safely used with the current PyTorch build."""

        if not torch.cuda.is_available():
            return

        try:
            self.device_name = torch.cuda.get_device_name(0)
        except Exception:
            self.device_name = "Unknown CUDA Device"

        try:
            self.device_capability = torch.cuda.get_device_capability(0)
        except Exception:
            self.device_capability = None

        try:
            # Attempt to allocate a trivial tensor on the GPU. If this fails we fall back to CPU.
            torch.zeros(1, device="cuda")
        except Exception as exc:
            warnings.warn(
                "CUDA is available but incompatible with the current PyTorch build. "
                "Falling back to CPU execution."
                f" (error: {exc})"
            )
            return

        self.device = torch.device("cuda")
        self.cuda_compatible = True

    def _select_dtype(self, preferred_dtype: torch.dtype) -> torch.dtype:
        """Return a dtype that is safe for the current device."""

        if self.cuda_compatible:
            return preferred_dtype

        # Many pipelines default to float16/bfloat16 for GPU usage. These dtypes are slower
        # or unsupported on CPU, so we promote them to float32 when we cannot rely on CUDA.
        if preferred_dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        return preferred_dtype
    
    def _cleanup_old_models(self):
        if len(self.loaded_models) >= self.max_loaded_models:
            sorted_models = sorted(self.model_last_used.items(), key=lambda x: x[1])
            oldest_model = sorted_models[0][0]
            print(f"Unloading {oldest_model} to make room for new model")
            self.unload_model(oldest_model)
    
    def _load_text_to_image_model(self, model_key: str):
        model_info = self.AVAILABLE_MODELS[model_key]
        model_id = model_info["model_id"]
        
        print(f"Loading {model_info['name']} from {model_id}...")

        skip_to_device = False

        if model_key == "flux":
            if not FLUX_AVAILABLE:
                raise ImportError("FluxPipeline not available")
            dtype = self._select_dtype(torch.bfloat16)
            low_vram_path = "/app/models/flux/unet/flux1-dev.safetensors"
            try:
                if hasattr(FluxPipeline, "from_single_file") and os.path.exists(low_vram_path):
                    pipe = FluxPipeline.from_single_file(
                        low_vram_path,
                        torch_dtype=dtype,
                        use_safetensors=True,
                        local_files_only=True,
                    )
                    print("  Loaded Flux from local low VRAM safetensors bundle")
                else:
                    raise ValueError("LowVRAM safetensor bundle not available")
            except Exception as exc:
                print(f"  LowVRAM load failed ({exc}); falling back to standard repo load")
                pipe = FluxPipeline.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    use_safetensors=True,
                )
        elif model_key == "sdxl":
            dtype = self._select_dtype(torch.float16)
            pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=dtype, use_safetensors=True, variant="fp16")
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        elif model_key == "sd3":
            dtype = self._select_dtype(torch.float16)
            pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=dtype, use_safetensors=True)
        elif model_key == "pony":
            dtype = self._select_dtype(torch.float16)
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                use_safetensors=True,
            )
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        else:
            raise ValueError(f"Unknown text-to-image model: {model_key}")

        if self.cuda_compatible:
            if model_key == "flux":
                pipe.enable_model_cpu_offload()
                if hasattr(pipe, "enable_vae_slicing"):
                    pipe.enable_vae_slicing()
                if hasattr(pipe, "enable_vae_tiling"):
                    pipe.enable_vae_tiling()
                if hasattr(pipe, "enable_sequential_cpu_offload"):
                    pipe.enable_sequential_cpu_offload()
                skip_to_device = True
                print("  Flux loaded with lowvram optimizations")
            else:
                pipe.enable_model_cpu_offload()
                # Only enable VAE slicing for pipelines that support it
                if hasattr(pipe, "enable_vae_slicing"):
                    pipe.enable_vae_slicing()
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                    print("  xformers enabled")
                except Exception:
                    pass

        if not skip_to_device:
            pipe = pipe.to(self.device)
        else:
            print("  Skipping explicit .to() since accelerate manages device placement")
        print(f"  {model_info['name']} loaded successfully")
        return pipe
    
    def _load_image_to_text_model(self, model_key: str) -> Tuple[Any, Any]:
        model_info = self.AVAILABLE_MODELS[model_key]
        model_id = model_info["model_id"]
        
        print(f"Loading {model_info['name']} from {model_id}...")
        
        device_map = "auto" if self.cuda_compatible else "cpu"
        dtype = self._select_dtype(torch.float16)

        if model_key == "llava":
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=dtype, device_map=device_map)
        elif model_key == "blip2":
            processor = Blip2Processor.from_pretrained(model_id)
            model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype, device_map=device_map)
        elif model_key == "qwen":
            processor = AutoProcessor.from_pretrained(model_id)
            model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype, device_map=device_map)
        else:
            raise ValueError(f"Unknown image-to-text model: {model_key}")

        print(f"  {model_info['name']} loaded successfully")
        return model, processor
    
    def _load_video_generation_model(self, model_key: str):
        model_info = self.AVAILABLE_MODELS[model_key]
        model_id = model_info["model_id"]

        print(f"Loading {model_info['name']} from {model_id}...")
        
        dtype = self._select_dtype(torch.float16)

        if model_key == "svd":
            pipe = StableVideoDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, variant="fp16")
            if self.cuda_compatible:
                pipe.enable_model_cpu_offload()

        elif model_key == "animatediff":
            # AnimateDiff Lightning
            adapter = MotionAdapter.from_pretrained("ByteDance/AnimateDiff-Lightning", torch_dtype=dtype)
            base_model = model_info.get("base_model", "emilianJR/epiCRealism")
            pipe = AnimateDiffPipeline.from_pretrained(base_model, motion_adapter=adapter, torch_dtype=dtype)
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            if self.cuda_compatible:
                pipe.enable_model_cpu_offload()
                pipe.enable_vae_slicing()

        elif model_key == "wan21":
            # WAN 2.1 proxy using CogVideoX (similar architecture)
            print("  ⚠️  WAN 2.1 Integration Note:")
            print("  Using CogVideoX-5b-I2V as WAN 2.1 proxy (similar DiT architecture)")
            print("  For native WAN 2.1 + LightX2V LoRA:")
            print("  → Use ComfyUI workflow: https://www.nextdiffusion.ai/tutorials/fast-image-to-video-comfyui-wan2-2-lightx2v-lora")
            print("  → Or LightX2V framework: https://github.com/ModelTC/LightX2V")
            
            if COGVIDEO_AVAILABLE:
                pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_id, torch_dtype=dtype)
                print("  Using CogVideoX-5b-I2V (fast image-to-video)")
            else:
                # Fallback to SVD
                print("  CogVideoX not available, using SVD as fallback")
                pipe = StableVideoDiffusionPipeline.from_pretrained(
                    "stabilityai/stable-video-diffusion-img2vid-xt",
                    torch_dtype=dtype,
                    variant="fp16"
                )

            if self.cuda_compatible:
                pipe.enable_model_cpu_offload()
                pipe.enable_vae_slicing()

        else:
            raise ValueError(f"Unknown video generation model: {model_key}")

        pipe = pipe.to(self.device)
        print(f"  {model_info['name']} loaded successfully")
        return pipe

    def _load_talking_head_model(self, model_key: str):
        """Load talking head models - now using hybrid approach with ComfyUI WAN 2.1"""
        
        model_info = self.AVAILABLE_MODELS[model_key]
        model_id = model_info["model_id"]

        print(f"Loading {model_info['name']} (Hybrid Mode)...")
        print("=" * 60)
        print("  HYBRID INFINITETALK")
        print("  Face Preprocessing: CPU (lightweight)")
        print("  Video Generation: ComfyUI WAN 2.1 (your working setup)")
        print("=" * 60)

        try:
            # Import hybrid InfiniteTalk wrapper
            from infinitetalk_hybrid import get_hybrid_infinitetalk_pipeline

            device_str = "cpu"  # Face preprocessing is CPU-based (lightweight)
            pipe = get_hybrid_infinitetalk_pipeline(device=device_str)

            print(f"  ✓ {model_info['name']} loaded successfully")
            print(f"  → Face detection: MediaPipe/OpenCV (CPU)")
            print(f"  → Video generation: ComfyUI WAN 2.1 (GPU via ComfyUI)")
            print(f"  → VRAM usage: ~5GB (only WAN 2.1, no duplication)")
            print(f"  → Generation speed: 30-60 seconds")
            print("=" * 60)
            
            return pipe

        except ImportError as e:
            print(f"  ❌ Error: infinitetalk_hybrid module not found")
            print(f"  → Make sure infinitetalk_hybrid.py is in your project root")
            print(f"  → Import error: {str(e)}")
            raise RuntimeError(
                f"Hybrid InfiniteTalk failed to load: {str(e)}. "
                "Ensure infinitetalk_hybrid.py is in your project directory."
            )
        except Exception as e:
            print(f"  ❌ Error loading Hybrid InfiniteTalk: {str(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(
                f"Hybrid InfiniteTalk failed to load: {str(e)}. "
                "Check logs above for details."
            )

    def _load_controlnet_model(self, model_key: str):
        model_info = self.AVAILABLE_MODELS[model_key]
        model_id = model_info["model_id"]

        print(f"Loading {model_info['name']} from {model_id}...")
        
        dtype = self._select_dtype(torch.float16)

        controlnet = ControlNetModel.from_pretrained(model_id, torch_dtype=dtype)
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_model_id, controlnet=controlnet, torch_dtype=dtype, variant="fp16"
        )

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        if self.cuda_compatible:
            pipe.enable_model_cpu_offload()
            pipe.enable_vae_slicing()

            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

        pipe = pipe.to(self.device)
        print(f"  {model_info['name']} loaded successfully")
        return pipe
    
    def load_model(self, model_key: str, model_type: str):
        if model_key not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_key}")
        
        if model_key in self.loaded_models:
            print(f"Using cached {model_key} model")
            self.model_last_used[model_key] = time.time()
            return self.loaded_models[model_key]
        
        self._cleanup_old_models()
        
        if model_type == "text-to-image":
            model = self._load_text_to_image_model(model_key)
        elif model_type == "image-to-text":
            model = self._load_image_to_text_model(model_key)
        elif model_type == "video-generation":
            model = self._load_video_generation_model(model_key)
        elif model_type == "controlnet":
            model = self._load_controlnet_model(model_key)
        elif model_type == "talking-head":
            model = self._load_talking_head_model(model_key)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.loaded_models[model_key] = model
        self.model_last_used[model_key] = time.time()
        
        if self.cuda_compatible:
            vram_used = torch.cuda.memory_allocated() / 1e9
            vram_cached = torch.cuda.memory_reserved() / 1e9
            print(f"  VRAM used: {vram_used:.2f} GB, cached: {vram_cached:.2f} GB")
        
        return model
    
    def unload_model(self, model_key: str):
        if model_key in self.loaded_models:
            print(f"Unloading {model_key}...")
            del self.loaded_models[model_key]
            del self.model_last_used[model_key]
            gc.collect()
            if self.cuda_compatible:
                torch.cuda.empty_cache()
                vram_used = torch.cuda.memory_allocated() / 1e9
                print(f"  VRAM after unload: {vram_used:.2f} GB")
    
    def unload_all(self):
        print("Unloading all models...")
        self.loaded_models.clear()
        self.model_last_used.clear()
        gc.collect()
        if self.cuda_compatible:
            torch.cuda.empty_cache()
            vram_used = torch.cuda.memory_allocated() / 1e9
            print(f"  VRAM after clearing all: {vram_used:.2f} GB")
    
    def get_loaded_models(self) -> list:
        return list(self.loaded_models.keys())
    
    def get_model_stats(self) -> Dict[str, Any]:
        stats = {}
        for model_key, last_used in self.model_last_used.items():
            time_since_used = time.time() - last_used
            stats[model_key] = {
                "loaded": True,
                "last_used_seconds_ago": round(time_since_used, 2),
                "vram_estimate_gb": self.AVAILABLE_MODELS[model_key]["vram_gb"]
            }
        return stats
