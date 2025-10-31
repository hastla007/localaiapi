"""InfiniteTalk inference wrapper.

This module provides a wrapper around InfiniteTalk for talking head generation.
InfiniteTalk generates videos from a face image and audio/text input.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union, List
from PIL import Image
import tempfile
import subprocess
import logging

logger = logging.getLogger(__name__)


class InfiniteTalkPipeline:
    """Wrapper for InfiniteTalk talking head generation."""
    
    def __init__(self, model_path: str = "MeiGen-AI/InfiniteTalk", device: str = "cuda"):
        """Initialize InfiniteTalk pipeline.
        
        Args:
            model_path: HuggingFace model path or local path
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device
        self.model_path = model_path
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load InfiniteTalk model components."""
        try:
            # Import InfiniteTalk modules
            from huggingface_hub import hf_hub_download, snapshot_download
            
            # Download model files
            logger.info(f"Downloading InfiniteTalk from {self.model_path}")
            model_dir = snapshot_download(
                repo_id=self.model_path,
                local_dir="/app/models/infinitetalk",
                local_dir_use_symlinks=False
            )
            
            logger.info(f"InfiniteTalk downloaded to {model_dir}")
            
            # Try to import and initialize InfiniteTalk
            try:
                # Check if InfiniteTalk package is installed
                import sys
                sys.path.insert(0, model_dir)
                
                # Import InfiniteTalk inference
                from inference import InfiniteTalkInference
                
                self.model = InfiniteTalkInference(
                    model_dir=model_dir,
                    device=self.device
                )
                logger.info("InfiniteTalk model loaded successfully")
                
            except ImportError as e:
                logger.warning(f"InfiniteTalk package not found: {e}")
                logger.info("Using fallback implementation")
                self.model = self._create_fallback_model(model_dir)
                
        except Exception as e:
            logger.error(f"Error loading InfiniteTalk: {e}")
            raise
    
    def _create_fallback_model(self, model_dir: str):
        """Create a fallback model implementation."""
        
        class FallbackInfiniteTalk:
            """Fallback implementation using available components."""
            
            def __init__(self, model_dir: str, device: str):
                self.model_dir = Path(model_dir)
                self.device = device
                
                # Load required models
                self._load_components()
            
            def _load_components(self):
                """Load model components."""
                try:
                    # Load face detection
                    import face_alignment
                    self.fa = face_alignment.FaceAlignment(
                        face_alignment.LandmarksType.TWO_D,
                        device=self.device
                    )
                    
                    # Load audio processor
                    import librosa
                    self.audio_processor = librosa
                    
                    logger.info("Fallback components loaded")
                    
                except Exception as e:
                    logger.error(f"Error loading fallback components: {e}")
                    raise
            
            def generate(
                self,
                image: Image.Image,
                audio_path: str,
                num_frames: int = 120,
                fps: int = 25,
                **kwargs
            ):
                """Generate talking head video."""
                # This is a placeholder - actual implementation would require
                # the full InfiniteTalk inference code
                raise NotImplementedError(
                    "InfiniteTalk inference not fully implemented. "
                    "Please install InfiniteTalk properly or use the official repository."
                )
        
        return FallbackInfiniteTalk(model_dir, self.device)
    
    def __call__(
        self,
        image: Union[Image.Image, str, Path],
        audio_path: Union[str, Path],
        num_frames: int = 120,
        fps: int = 25,
        expression_scale: float = 1.0,
        head_motion_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
        **kwargs
    ) -> List[Image.Image]:
        """Generate talking head video.
        
        Args:
            image: Input face image
            audio_path: Path to audio file
            num_frames: Number of frames to generate
            fps: Frames per second
            expression_scale: Expression intensity (0.5-2.0)
            head_motion_scale: Head movement intensity (0.5-2.0)
            generator: Random generator for reproducibility
            
        Returns:
            List of PIL Images (video frames)
        """
        # Convert image to PIL if needed
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be PIL Image, str, or Path")
        
        # Save image temporarily if needed
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
            image.save(tmp_img.name)
            image_path = tmp_img.name
        
        try:
            # Call model inference
            if hasattr(self.model, 'generate'):
                frames = self.model.generate(
                    image=image,
                    audio_path=str(audio_path),
                    num_frames=num_frames,
                    fps=fps,
                    expression_scale=expression_scale,
                    head_motion_scale=head_motion_scale,
                    **kwargs
                )
            else:
                raise NotImplementedError(
                    "InfiniteTalk model not properly initialized. "
                    "Please check installation."
                )
            
            return frames
            
        finally:
            # Cleanup temp file
            Path(image_path).unlink(missing_ok=True)
    
    def to(self, device: str):
        """Move model to device."""
        self.device = device
        if hasattr(self.model, 'to'):
            self.model.to(device)
        return self


def get_infinitetalk_pipeline(device: str = "cuda") -> InfiniteTalkPipeline:
    """Get InfiniteTalk pipeline instance.
    
    Args:
        device: Device to run on
        
    Returns:
        InfiniteTalkPipeline instance
    """
    return InfiniteTalkPipeline(device=device)
