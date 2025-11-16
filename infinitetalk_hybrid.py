"""Hybrid InfiniteTalk using ComfyUI WAN 2.1 for video generation

This approach:
1. Uses face detection/preprocessing (simple OpenCV/MediaPipe)
2. Uses your TTS server for audio
3. Uses your working ComfyUI WAN 2.1 for video generation
"""

import torch
import tempfile
import logging
from pathlib import Path
from typing import Optional, List
from PIL import Image
import numpy as np
import subprocess

logger = logging.getLogger(__name__)


class HybridInfiniteTalkPipeline:
    """Simplified talking head using face prep + ComfyUI WAN 2.1"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.initialized = False
        
    def _ensure_face_detection(self):
        """Lazy load face detection only when needed"""
        if self.initialized:
            return
            
        try:
            # Try MediaPipe first (lightweight, no models to download)
            import mediapipe as mp
            self.face_detection = mp.solutions.face_detection.FaceDetection(
                min_detection_confidence=0.5
            )
            self.use_mediapipe = True
            logger.info("✓ Using MediaPipe for face detection")
        except ImportError:
            logger.warning("MediaPipe not available, using basic preprocessing")
            self.face_detection = None
            self.use_mediapipe = False
            
        self.initialized = True
    
    def _preprocess_face(self, image: Image.Image) -> Image.Image:
        """Preprocess face image for optimal talking head results

        - Detect and center face
        - Crop to portrait orientation
        - Resize to optimal dimensions
        """
        self._ensure_face_detection()

        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Target sizes for WAN 2.1 (portrait orientation)
        target_width = 576
        target_height = 1024

        # Validate target dimensions to prevent division by zero
        if target_width <= 0 or target_height <= 0:
            raise ValueError(f"Invalid target dimensions: {target_width}x{target_height}")
        
        if self.use_mediapipe and self.face_detection:
            try:
                # Convert PIL to numpy for MediaPipe
                image_np = np.array(image)
                results = self.face_detection.process(image_np)
                
                if results.detections:
                    # Get first detected face
                    detection = results.detections[0]
                    bboxC = detection.location_data.relative_bounding_box
                    
                    ih, iw, _ = image_np.shape
                    
                    # Convert relative coordinates to absolute
                    x = int(bboxC.xmin * iw)
                    y = int(bboxC.ymin * ih)
                    w = int(bboxC.width * iw)
                    h = int(bboxC.height * ih)
                    
                    # Expand bbox for portrait framing (more headroom)
                    expansion = 0.5  # 50% expansion for shoulders/headroom
                    x_expanded = x - int(w * expansion / 2)
                    y_expanded = y - int(h * expansion)
                    w_expanded = int(w * (1 + expansion))
                    h_expanded = int(h * (1 + expansion * 1.5))

                    # Validate expanded dimensions to prevent division by zero
                    if w_expanded <= 0 or h_expanded <= 0:
                        logger.warning(f"Invalid expanded dimensions: {w_expanded}x{h_expanded}, using center crop")
                        image = self._center_crop_portrait(image, target_width, target_height)
                    else:
                        # Ensure portrait aspect ratio before bounds checking
                        aspect_ratio = target_height / target_width
                        if h_expanded / w_expanded < aspect_ratio:
                            # Too wide, increase height
                            new_h = int(w_expanded * aspect_ratio)
                            y_expanded = y_expanded - (new_h - h_expanded) // 2
                            h_expanded = new_h
                        else:
                            # Too tall, increase width
                            new_w = int(h_expanded / aspect_ratio)
                            x_expanded = x_expanded - (new_w - w_expanded) // 2
                            w_expanded = new_w

                        # Ensure bounds - ensure we don't go outside image dimensions
                        # If expanded box exceeds image boundaries, shift and adjust
                        if x_expanded < 0:
                            x_expanded = 0
                        if y_expanded < 0:
                            y_expanded = 0
                        if x_expanded + w_expanded > iw:
                            # If too wide, try to shift left first
                            overflow = (x_expanded + w_expanded) - iw
                            x_expanded = max(0, x_expanded - overflow)
                            # If still too wide, reduce width
                            w_expanded = min(w_expanded, iw - x_expanded)
                        if y_expanded + h_expanded > ih:
                            # If too tall, try to shift up first
                            overflow = (y_expanded + h_expanded) - ih
                            y_expanded = max(0, y_expanded - overflow)
                            # If still too tall, reduce height
                            h_expanded = min(h_expanded, ih - y_expanded)

                        # Final sanity check - ensure we have valid dimensions
                        if w_expanded <= 0 or h_expanded <= 0:
                            logger.warning(f"Invalid crop dimensions: {w_expanded}x{h_expanded}, using center crop")
                            image = self._center_crop_portrait(image, target_width, target_height)
                        else:
                            # Crop to face region
                            image = image.crop((x_expanded, y_expanded, x_expanded + w_expanded, y_expanded + h_expanded))
                            logger.info(f"✓ Face detected and cropped: {w_expanded}x{h_expanded}")
                else:
                    logger.info("No face detected, using center crop")
                    # Fallback to center crop
                    image = self._center_crop_portrait(image, target_width, target_height)
                    
            except Exception as e:
                logger.warning(f"Face detection failed: {e}, using center crop")
                image = self._center_crop_portrait(image, target_width, target_height)
        else:
            # No face detection, use smart center crop
            image = self._center_crop_portrait(image, target_width, target_height)
        
        # Final resize to target dimensions
        if image.size != (target_width, target_height):
            image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            logger.info(f"✓ Resized to {target_width}x{target_height}")
        
        return image
    
    def _center_crop_portrait(self, image: Image.Image, target_w: int, target_h: int) -> Image.Image:
        """Smart center crop to portrait orientation"""
        w, h = image.size

        # Validate dimensions to prevent division by zero
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid image dimensions: {w}x{h}")
        if target_w <= 0 or target_h <= 0:
            raise ValueError(f"Invalid target dimensions: {target_w}x{target_h}")

        aspect_ratio = target_h / target_w

        # Calculate crop dimensions to match aspect ratio
        if h / w < aspect_ratio:
            # Image is too wide, crop width
            new_w = int(h / aspect_ratio)
            left = (w - new_w) // 2
            image = image.crop((left, 0, left + new_w, h))
        else:
            # Image is too tall, crop height
            new_h = int(w * aspect_ratio)
            top = (h - new_h) // 3  # Crop more from bottom (favor top of frame)
            image = image.crop((0, top, w, top + new_h))
        
        return image
    
    def __call__(
        self,
        image: Image.Image,
        audio_path: str,
        num_frames: int = 120,
        fps: int = 25,
        expression_scale: float = 1.0,  # Not used directly, but kept for API compatibility
        head_motion_scale: float = 1.0,  # Not used directly, but kept for API compatibility
        generator: Optional[torch.Generator] = None,
        **kwargs
    ) -> str:
        """Generate talking head video using preprocessed face + ComfyUI WAN 2.1
        
        Returns:
            Path to generated video file
        """
        
        logger.info("=== Hybrid InfiniteTalk: Face Prep + WAN 2.1 ===")
        
        # Step 1: Preprocess face image
        logger.info("Step 1: Preprocessing face image...")
        preprocessed_face = self._preprocess_face(image)
        
        # Step 2: Save preprocessed image to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_face:
            face_path = Path(tmp_face.name)
            preprocessed_face.save(face_path)
            logger.info(f"✓ Face image saved: {face_path}")
        
        # Step 3: Return preprocessed face path
        # The actual video generation will be handled by ComfyUI WAN 2.1
        # in the main.py endpoint
        return str(face_path)
    
    def to(self, device: str):
        """Move model to device (compatibility method)"""
        self.device = device
        return self


def get_hybrid_infinitetalk_pipeline(device: str = "cuda") -> HybridInfiniteTalkPipeline:
    """Get hybrid InfiniteTalk pipeline instance"""
    return HybridInfiniteTalkPipeline(device=device)
