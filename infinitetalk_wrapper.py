"""InfiniteTalk inference wrapper - Lazy Loading"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, List
from PIL import Image
import logging
import sys
import os

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
                logger.info("InfiniteTalk not found. Downloading from GitHub...")
                import subprocess
                subprocess.run(
                    ["git", "clone", "https://github.com/MeiGen-AI/InfiniteTalk.git", str(self.repo_path)],
                    check=True,
                    capture_output=True
                )
                logger.info("✓ InfiniteTalk repository cloned")
            
            # Download model checkpoints if not exists
            checkpoint_dir = self.repo_path / "checkpoints"
            if not checkpoint_dir.exists() or not any(checkpoint_dir.glob("*.pth")):
                logger.info("Downloading InfiniteTalk model checkpoints...")
                self._download_checkpoints()
                logger.info("✓ Model checkpoints downloaded")
            
            # Add to Python path
            sys.path.insert(0, str(self.repo_path))
            
            # Import and initialize
            from inference import InfiniteTalkInference
            
            self.model = InfiniteTalkInference(
                checkpoint_path=str(checkpoint_dir),
                device=self.device
            )
            logger.info("✓ InfiniteTalk model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup InfiniteTalk: {e}")
            raise RuntimeError(
                f"InfiniteTalk setup failed: {e}\n"
                "This will download automatically on first use.\n"
                "Ensure you have internet connection and git installed."
            )
    
    def _download_checkpoints(self):
        """Download model checkpoints from HuggingFace."""
        from huggingface_hub import snapshot_download
        
        try:
            snapshot_download(
                repo_id="MeiGen-AI/InfiniteTalk",
                local_dir=str(self.repo_path),
                local_dir_use_symlinks=False,
                allow_patterns=["checkpoints/*", "*.py", "*.json"]
            )
        except Exception as e:
            logger.warning(f"HuggingFace download failed: {e}")
            logger.info("Trying alternative download method...")
            
            # Fallback: download individual files
            from huggingface_hub import hf_hub_download
            checkpoint_dir = self.repo_path / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # List of critical files (adjust based on actual InfiniteTalk structure)
            files = [
                "checkpoints/infinitetalk_base.pth",
                "checkpoints/audio_encoder.pth", 
                "checkpoints/face_encoder.pth",
            ]
            
            for file in files:
                try:
                    hf_hub_download(
                        repo_id="MeiGen-AI/InfiniteTalk",
                        filename=file,
                        local_dir=str(self.repo_path),
                        local_dir_use_symlinks=False
                    )
                except Exception as file_error:
                    logger.warning(f"Could not download {file}: {file_error}")
    
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
        """
        # Lazy load on first use
        self._ensure_installed()
        
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Preprocess image
        image = self._preprocess_image(image)
        
        try:
            # Generate frames
            frames = self.model.generate(
                face_image=image,
                audio_path=audio_path,
                num_frames=num_frames,
                fps=fps,
                expression_scale=expression_scale,
                head_motion_scale=head_motion_scale,
            )
            
            # Convert to PIL Images
            if isinstance(frames, torch.Tensor):
                frames = self._tensor_to_pil(frames)
            elif isinstance(frames, np.ndarray):
                frames = [Image.fromarray(f) for f in frames]
            
            logger.info(f"Generated {len(frames)} frames")
            return frames
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Video generation failed: {e}")
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess face image."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to 512x512 (typical for face models)
        if image.size != (512, 512):
            image = image.resize((512, 512), Image.Resampling.LANCZOS)
        
        return image
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> List[Image.Image]:
        """Convert tensor to PIL Images."""
        if tensor.dim() == 4:
            if tensor.shape[1] == 3:  # [N, C, H, W]
                tensor = tensor.permute(0, 2, 3, 1)  # [N, H, W, C]
            
            frames = []
            for frame in tensor:
                frame_np = frame.cpu().numpy()
                frame_np = np.clip(frame_np * 255, 0, 255).astype(np.uint8)
                frames.append(Image.fromarray(frame_np))
            
            return frames
        else:
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
    
    def to(self, device: str):
        """Move model to device."""
        self.device = device
        if self.model and hasattr(self.model, 'to'):
            self.model.to(device)
        return self


def get_infinitetalk_pipeline(device: str = "cuda") -> InfiniteTalkPipeline:
    """Get InfiniteTalk pipeline instance."""
    return InfiniteTalkPipeline(device=device)
