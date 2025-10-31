"""InfiniteTalk inference wrapper - Lazy Loading with proper repo handling"""

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
    """Lazy-loading wrapper for InfiniteTalk."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.base_path = Path("/app/models/infinitetalk")
        self.repo_path = self.base_path / "InfiniteTalk"
        
    def _ensure_installed(self):
        """Download and setup InfiniteTalk on first use."""
        if self.model is not None:
            return  # Already loaded
        
        try:
            # Create base directory
            self.base_path.mkdir(parents=True, exist_ok=True)
            
            # Clone repository if not exists
            if not self.repo_path.exists():
                logger.info("InfiniteTalk not found. Cloning from GitHub...")
                subprocess.run(
                    ["git", "clone", "https://github.com/MeiGen-AI/InfiniteTalk.git", str(self.repo_path)],
                    check=True,
                    capture_output=True,
                    text=True
                )
                logger.info("✓ InfiniteTalk repository cloned")
            
            # Add to Python path
            if str(self.repo_path) not in sys.path:
                sys.path.insert(0, str(self.repo_path))
            
            # Download model checkpoints if needed
            self._ensure_checkpoints()
            
            # Import InfiniteTalk modules
            logger.info("Importing InfiniteTalk modules...")
            from src.pipelines.talking_head_pipeline import TalkingHeadPipeline
            from src.models.audio_encoder import AudioEncoder
            from src.models.face_encoder import FaceEncoder
            
            # Initialize pipeline
            logger.info("Initializing InfiniteTalk pipeline...")
            self.model = TalkingHeadPipeline(
                device=self.device,
                checkpoint_dir=str(self.repo_path / "checkpoints")
            )
            
            logger.info("✓ InfiniteTalk model loaded successfully")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Git clone failed: {e.stderr}")
            raise RuntimeError(
                f"Failed to clone InfiniteTalk repository: {e.stderr}\n"
                "Ensure git is installed and you have internet connection."
            )
        except ImportError as e:
            logger.error(f"Import failed: {e}")
            raise RuntimeError(
                f"Failed to import InfiniteTalk modules: {e}\n"
                "The repository structure may have changed. Check GitHub for updates."
            )
        except Exception as e:
            logger.error(f"InfiniteTalk setup failed: {e}")
            raise RuntimeError(f"InfiniteTalk setup failed: {e}")
    
    def _ensure_checkpoints(self):
        """Download model checkpoints from HuggingFace if not present."""
        checkpoint_dir = self.repo_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if checkpoints exist
        required_files = [
            "infinitetalk_base.pth",
            "audio_processor.pth",
            "motion_generator.pth"
        ]
        
        missing_files = [f for f in required_files if not (checkpoint_dir / f).exists()]
        
        if not missing_files:
            logger.info("✓ Model checkpoints already present")
            return
        
        logger.info("Downloading InfiniteTalk checkpoints from HuggingFace...")
        
        try:
            from huggingface_hub import snapshot_download
            
            snapshot_download(
                repo_id="MeiGen-AI/InfiniteTalk",
                local_dir=str(self.repo_path),
                local_dir_use_symlinks=False,
                allow_patterns=["checkpoints/*", "*.py", "*.json", "*.yaml"]
            )
            
            logger.info("✓ Checkpoints downloaded successfully")
            
        except Exception as e:
            logger.warning(f"Snapshot download failed: {e}")
            logger.info("Attempting individual file downloads...")
            
            # Fallback: Try downloading specific files
            from huggingface_hub import hf_hub_download
            
            for filename in required_files:
                try:
                    file_path = f"checkpoints/{filename}"
                    logger.info(f"Downloading {filename}...")
                    hf_hub_download(
                        repo_id="MeiGen-AI/InfiniteTalk",
                        filename=file_path,
                        local_dir=str(self.repo_path),
                        local_dir_use_symlinks=False
                    )
                    logger.info(f"✓ Downloaded {filename}")
                except Exception as file_error:
                    logger.error(f"Failed to download {filename}: {file_error}")
                    raise RuntimeError(
                        f"Could not download required checkpoint {filename}. "
                        "Check your internet connection and HuggingFace access."
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
        """Generate talking head video.
        
        Downloads InfiniteTalk on first call.
        
        Args:
            image: Face image (PIL Image)
            audio_path: Path to audio file (WAV/MP3)
            num_frames: Number of frames to generate
            fps: Frames per second
            expression_scale: Expression intensity (0.5-2.0)
            head_motion_scale: Head movement intensity (0.5-2.0)
            generator: Optional random generator for reproducibility
            
        Returns:
            List of PIL Images (video frames)
        """
        # Lazy load on first use
        self._ensure_installed()
        
        # Validate audio file
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Preprocess image
        image = self._preprocess_image(image)
        
        try:
            logger.info(f"Generating {num_frames} frames at {fps} FPS...")
            
            # Generate frames using InfiniteTalk
            output = self.model(
                face_image=image,
                audio_path=str(audio_path),
                num_frames=num_frames,
                fps=fps,
                expression_scale=expression_scale,
                head_motion_scale=head_motion_scale,
            )
            
            # Convert output to list of PIL Images
            if isinstance(output, dict) and 'frames' in output:
                frames = output['frames']
            elif hasattr(output, 'frames'):
                frames = output.frames
            else:
                frames = output
            
            # Ensure frames are PIL Images
            frames = self._convert_to_pil(frames)
            
            logger.info(f"✓ Generated {len(frames)} frames successfully")
            return frames
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Video generation failed: {e}")
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess face image to proper format and size."""
        # Convert to RGB if needed
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
        if isinstance(frames, torch.Tensor):
            # Tensor format: [N, C, H, W] or [N, H, W, C]
            frames = frames.cpu().numpy()
        
        if isinstance(frames, np.ndarray):
            # Numpy array
            if frames.ndim == 4:
                # [N, H, W, C] or [N, C, H, W]
                if frames.shape[1] == 3 or frames.shape[1] == 4:
                    # [N, C, H, W] -> [N, H, W, C]
                    frames = np.transpose(frames, (0, 2, 3, 1))
                
                for frame in frames:
                    # Normalize to 0-255 if needed
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
            # Already a list, check if PIL Images
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
