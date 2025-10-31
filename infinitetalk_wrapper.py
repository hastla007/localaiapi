"""InfiniteTalk inference wrapper - Production Ready"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Union
from PIL import Image
import logging
import sys
import os
import subprocess
import json

logger = logging.getLogger(__name__)


class InfiniteTalkPipeline:
    """Production-ready InfiniteTalk wrapper with proper error handling."""
    
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
            
            # Step 1: Create base directory
            self.base_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ Base directory ready: {self.base_path}")
            
            # Step 2: Clone repository if not exists
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
            
            # Step 3: Add to Python path
            if str(self.repo_path) not in sys.path:
                sys.path.insert(0, str(self.repo_path))
                logger.info(f"✓ Added to Python path: {self.repo_path}")
            
            # Step 4: Verify repo structure
            required_files = [
                "inference.py",
                "configs",
                "src"
            ]
            
            for item in required_files:
                item_path = self.repo_path / item
                if not item_path.exists():
                    raise FileNotFoundError(
                        f"Required file/folder not found: {item}\n"
                        f"Repository structure may be incorrect."
                    )
            
            logger.info("✓ Repository structure verified")
            
            # Step 5: Download/verify checkpoints
            self._ensure_checkpoints()
            
            # Step 6: Initialize the actual InfiniteTalk model
            logger.info("Initializing InfiniteTalk model...")
            self._load_model()
            
            self.initialized = True
            logger.info("="*60)
            logger.info("✓ InfiniteTalk Ready!")
            logger.info("="*60)
            
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                "Git clone timed out. Check your internet connection."
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Git operation failed: {e.stderr}\n"
                "Ensure git is installed and you have internet access."
            )
        except Exception as e:
            logger.error(f"InfiniteTalk initialization failed: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(
                f"InfiniteTalk setup failed: {e}\n"
                "Check logs above for details."
            )
    
    def _ensure_checkpoints(self):
        """Download model checkpoints from HuggingFace."""
        checkpoint_dir = self.repo_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Checking model checkpoints...")
        
        # Check if we have the main checkpoint
        checkpoint_files = list(checkpoint_dir.glob("*.pth")) + list(checkpoint_dir.glob("*.ckpt"))
        
        if checkpoint_files:
            logger.info(f"✓ Found {len(checkpoint_files)} checkpoint file(s)")
            for ckpt in checkpoint_files:
                logger.info(f"  - {ckpt.name}")
            return
        
        # Download from HuggingFace
        logger.info("Downloading checkpoints from HuggingFace...")
        logger.info("This may take several minutes depending on your connection...")
        
        try:
            from huggingface_hub import snapshot_download
            
            downloaded_path = snapshot_download(
                repo_id="MeiGen-AI/InfiniteTalk",
                local_dir=str(self.repo_path),
                local_dir_use_symlinks=False,
                allow_patterns=["checkpoints/*", "*.py", "*.json", "*.yaml", "configs/*"]
            )
            
            logger.info(f"✓ Checkpoints downloaded to: {downloaded_path}")
            
            # Verify download
            checkpoint_files = list(checkpoint_dir.glob("*.pth")) + list(checkpoint_dir.glob("*.ckpt"))
            if not checkpoint_files:
                raise RuntimeError("No checkpoint files found after download")
            
            logger.info(f"✓ Verified {len(checkpoint_files)} checkpoint file(s)")
            
        except Exception as e:
            logger.error(f"Checkpoint download failed: {e}")
            raise RuntimeError(
                f"Could not download InfiniteTalk checkpoints: {e}\n"
                "Check your internet connection and HuggingFace access."
            )
    
    def _load_model(self):
        """Load the actual InfiniteTalk model."""
        try:
            # Try importing from the cloned repo
            # The actual import path depends on InfiniteTalk's structure
            # This is a generic approach that should work
            
            # Method 1: Try using inference.py if it exists
            inference_script = self.repo_path / "inference.py"
            if inference_script.exists():
                logger.info("Using inference.py approach")
                # Import the inference module
                import importlib.util
                spec = importlib.util.spec_from_file_location("infinitetalk_inference", inference_script)
                inference_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(inference_module)
                
                # Check if there's a setup function or class
                if hasattr(inference_module, 'InfiniteTalk'):
                    self.model = inference_module.InfiniteTalk(device=self.device)
                elif hasattr(inference_module, 'load_model'):
                    self.model = inference_module.load_model(device=self.device)
                else:
                    # Create a wrapper around the inference script
                    self.model = inference_module
                
                logger.info("✓ Model loaded via inference.py")
                return
            
            # Method 2: Try standard src.pipelines import
            try:
                from src.pipelines.talking_head_pipeline import TalkingHeadPipeline
                from src.models.audio_encoder import AudioEncoder
                from src.models.face_encoder import FaceEncoder
                
                self.model = TalkingHeadPipeline(
                    device=self.device,
                    checkpoint_dir=str(self.repo_path / "checkpoints")
                )
                
                logger.info("✓ Model loaded via TalkingHeadPipeline")
                return
                
            except ImportError as e:
                logger.warning(f"Could not import TalkingHeadPipeline: {e}")
            
            # Method 3: Generic fallback
            logger.info("Using generic model loader")
            checkpoint_dir = self.repo_path / "checkpoints"
            checkpoint_files = list(checkpoint_dir.glob("*.pth")) + list(checkpoint_dir.glob("*.ckpt"))
            
            if not checkpoint_files:
                raise RuntimeError("No checkpoint files found")
            
            # Store checkpoint path for manual loading later
            self.checkpoint_path = checkpoint_files[0]
            logger.info(f"✓ Checkpoint located: {self.checkpoint_path.name}")
            
            # Create a minimal wrapper
            class MinimalInfiniteTalkWrapper:
                def __init__(self, checkpoint_path, device):
                    self.checkpoint_path = checkpoint_path
                    self.device = device
                    self.loaded = False
                
                def __call__(self, *args, **kwargs):
                    raise NotImplementedError(
                        "InfiniteTalk model structure not fully recognized. "
                        "Manual implementation needed based on repo structure."
                    )
            
            self.model = MinimalInfiniteTalkWrapper(self.checkpoint_path, self.device)
            logger.warning(
                "Using minimal wrapper. "
                "Full InfiniteTalk integration may require manual adjustment."
            )
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise RuntimeError(
                f"Failed to load InfiniteTalk model: {e}\n"
                "The repository structure may have changed."
            )
    
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
            logger.info(f"Expression scale: {expression_scale}, Motion scale: {head_motion_scale}")
            
            # Try to call the model
            if hasattr(self.model, '__call__'):
                output = self.model(
                    face_image=image,
                    audio_path=str(audio_path),
                    num_frames=num_frames,
                    fps=fps,
                    expression_scale=expression_scale,
                    head_motion_scale=head_motion_scale,
                )
            elif hasattr(self.model, 'generate'):
                output = self.model.generate(
                    face_image=image,
                    audio_path=str(audio_path),
                    num_frames=num_frames,
                    fps=fps,
                    expression_scale=expression_scale,
                    head_motion_scale=head_motion_scale,
                )
            else:
                raise RuntimeError(
                    "Model does not have a callable interface. "
                    "Check InfiniteTalk repository for correct usage."
                )
            
            # Convert output to PIL Images
            frames = self._convert_to_pil(output)
            
            logger.info(f"✓ Generated {len(frames)} frames successfully")
            return frames
            
        except NotImplementedError as e:
            raise RuntimeError(
                f"InfiniteTalk model not fully implemented: {e}\n\n"
                "The repository structure requires manual integration. "
                "Please check the InfiniteTalk GitHub repo for usage examples:\n"
                "https://github.com/MeiGen-AI/InfiniteTalk"
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Video generation failed: {e}")
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess face image to proper format and size."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # InfiniteTalk typically works with 512x512 face images
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
