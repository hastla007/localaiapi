"""InfiniteTalk inference wrapper - Complete working implementation"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, List
from PIL import Image
import logging
import sys
import os
import subprocess

logger = logging.getLogger(__name__)


class InfiniteTalkPipeline:
    """Production-ready InfiniteTalk wrapper."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.base_path = Path("/app/models/infinitetalk")
        self.repo_path = self.base_path / "InfiniteTalk"
        self.initialized = False
        
    def _ensure_installed(self):
        """Download and setup InfiniteTalk on first use."""
        if self.initialized:
            return
        
        try:
            logger.info("="*60)
            logger.info("InfiniteTalk Initialization Starting")
            logger.info("="*60)
            
            # Create base directory
            self.base_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ Base directory ready: {self.base_path}")
            
            # Clone repository if not exists
            if not self.repo_path.exists():
                logger.info("Cloning InfiniteTalk from GitHub...")
                result = subprocess.run(
                    [
                        "git", "clone",
                        "https://github.com/MeiGen-AI/InfiniteTalk.git",
                        str(self.repo_path)
                    ],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode != 0:
                    raise RuntimeError(f"Git clone failed: {result.stderr}")
                
                logger.info("✓ Repository cloned successfully")
            else:
                logger.info("✓ Repository already exists")
            
            # Add to Python path
            if str(self.repo_path) not in sys.path:
                sys.path.insert(0, str(self.repo_path))
                logger.info(f"✓ Added to Python path: {self.repo_path}")
            
            # Install requirements
            self._install_requirements()
            
            # Download checkpoints
            self._ensure_checkpoints()
            
            # Load model
            self._load_model()
            
            self.initialized = True
            logger.info("="*60)
            logger.info("✓ InfiniteTalk Ready!")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"InfiniteTalk initialization failed: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"InfiniteTalk setup failed: {e}")
    
    def _install_requirements(self):
        """Install InfiniteTalk-specific requirements."""
        logger.info("Installing InfiniteTalk requirements...")
        
        req_file = self.repo_path / "requirements.txt"
        if req_file.exists():
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", str(req_file), "--no-cache-dir"],
                    check=True,
                    capture_output=True,
                    text=True
                )
                logger.info("✓ Requirements installed")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Some requirements failed to install: {e.stderr}")
                # Continue anyway - core deps might already be installed
    
    def _ensure_checkpoints(self):
        """Download model checkpoints from HuggingFace."""
        from huggingface_hub import hf_hub_download
        
        checkpoint_dir = self.repo_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Checking model checkpoints...")
        
        # Define required checkpoints
        required_files = [
            "audio_processor.pth",
            "face_encoder.pth", 
            "generator.pth",
            "config.yaml"
        ]
        
        try:
            for filename in required_files:
                local_path = checkpoint_dir / filename
                if not local_path.exists():
                    logger.info(f"Downloading {filename}...")
                    hf_hub_download(
                        repo_id="MeiGen-AI/InfiniteTalk",
                        filename=f"checkpoints/{filename}",
                        local_dir=str(self.repo_path),
                        local_dir_use_symlinks=False
                    )
            
            logger.info("✓ All checkpoints ready")
            
        except Exception as e:
            logger.error(f"Checkpoint download failed: {e}")
            raise RuntimeError(f"Could not download checkpoints: {e}")
    
    def _load_model(self):
        """Load the InfiniteTalk model."""
        try:
            # Import InfiniteTalk modules
            from inference import InfiniteTalkInference
            
            checkpoint_path = str(self.repo_path / "checkpoints")
            
            self.model = InfiniteTalkInference(
                checkpoint_dir=checkpoint_path,
                device=self.device
            )
            
            logger.info("✓ Model loaded successfully")
            
        except ImportError:
            # Fallback: try alternative import
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "inference", 
                    self.repo_path / "inference.py"
                )
                inference_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(inference_module)
                
                checkpoint_path = str(self.repo_path / "checkpoints")
                self.model = inference_module.InfiniteTalkInference(
                    checkpoint_dir=checkpoint_path,
                    device=self.device
                )
                
                logger.info("✓ Model loaded via fallback method")
                
            except Exception as e:
                logger.error(f"Could not load model: {e}")
                raise RuntimeError(f"Failed to load InfiniteTalk model: {e}")
    
    def __call__(
        self,
        image: Image.Image,
        audio_path: str,
        num_frames: int = 120,
        fps: int = 25,
        expression_scale: float = 1.0,
        head_motion_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
        **kwargs
    ) -> List[Image.Image]:
        """Generate talking head video."""
        
        # Ensure initialized
        self._ensure_installed()
        
        # Validate inputs
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Preprocess image
        image = self._preprocess_image(image)
        
        try:
            logger.info(f"Generating {num_frames} frames at {fps} FPS...")
            
            # Call InfiniteTalk model
            output = self.model.generate(
                face_image=np.array(image),
                audio_path=str(audio_path),
                num_frames=num_frames,
                fps=fps,
                expression_scale=expression_scale,
                head_motion_scale=head_motion_scale,
            )
            
            # Convert output to PIL Images
            frames = self._convert_to_pil(output)
            
            logger.info(f"✓ Generated {len(frames)} frames successfully")
            return frames
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Video generation failed: {e}")
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess face image to proper format and size."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # InfiniteTalk works with 512x512 face images
        target_size = (512, 512)
        if image.size != target_size:
            logger.info(f"Resizing image from {image.size} to {target_size}")
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        return image
    
    def _convert_to_pil(self, frames) -> List[Image.Image]:
        """Convert various frame formats to list of PIL Images."""
        pil_frames = []
        
        # Handle different output formats
        if isinstance(frames, dict):
            if 'frames' in frames:
                frames = frames['frames']
            elif 'images' in frames:
                frames = frames['images']
        
        if hasattr(frames, 'frames'):
            frames = frames.frames
        
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()
        
        if isinstance(frames, np.ndarray):
            if frames.ndim == 4:
                # [N, C, H, W] or [N, H, W, C]
                if frames.shape[1] == 3 or frames.shape[1] == 4:
                    # [N, C, H, W] -> [N, H, W, C]
                    frames = np.transpose(frames, (0, 2, 3, 1))
                
                for frame in frames:
                    # Normalize to 0-255
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    else:
                        frame = np.clip(frame, 0, 255).astype(np.uint8)
                    
                    # Handle RGBA -> RGB
                    if frame.shape[-1] == 4:
                        frame = frame[..., :3]
                    
                    pil_frames.append(Image.fromarray(frame))
            else:
                raise ValueError(f"Unexpected numpy array shape: {frames.shape}")
        
        elif isinstance(frames, list):
            for frame in frames:
                if isinstance(frame, Image.Image):
                    pil_frames.append(frame)
                elif isinstance(frame, np.ndarray):
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    pil_frames.append(Image.fromarray(frame))
                else:
                    raise ValueError(f"Unexpected frame type in list: {type(frame)}")
        else:
            raise ValueError(f"Unexpected frames type: {type(frames)}")
        
        return pil_frames
    
    def to(self, device: str):
        """Move model to device."""
        self.device = device
        if self.model and hasattr(self.model, 'to'):
            self.model.to(device)
        return self


def get_infinitetalk_pipeline(device: str = "cuda") -> InfiniteTalkPipeline:
    """Get InfiniteTalk pipeline instance."""
    return InfiniteTalkPipeline(device=device)
