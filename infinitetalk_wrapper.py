"""InfiniteTalk inference wrapper - Subprocess-based implementation for real InfiniteTalk"""

import torch
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Optional, List
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class InfiniteTalkPipeline:
    """Production-ready InfiniteTalk wrapper using subprocess calls."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
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
            
            # Install requirements
            self._install_requirements()
            
            # Download model checkpoints
            self._ensure_checkpoints()
            
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
                    ["pip", "install", "-r", str(req_file), "--break-system-packages"],
                    check=True,
                    capture_output=True,
                    text=True,
                    cwd=str(self.repo_path)
                )
                logger.info("✓ Requirements installed")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Some requirements failed: {e.stderr}")
    
    def _ensure_checkpoints(self):
        """Download model checkpoints using huggingface-cli."""
        logger.info("Downloading InfiniteTalk model checkpoints...")
        
        weights_dir = self.base_path / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoints = [
            ("Wan-AI/Wan2.1-I2V-14B-480P", "Wan2.1-I2V-14B-480P"),
            ("TencentGameMate/chinese-wav2vec2-base", "chinese-wav2vec2-base"),
            ("MeiGen-AI/InfiniteTalk", "InfiniteTalk"),
        ]
        
        for repo_id, local_name in checkpoints:
            local_dir = weights_dir / local_name
            if not local_dir.exists():
                logger.info(f"Downloading {repo_id}...")
                try:
                    subprocess.run(
                        [
                            "huggingface-cli", "download",
                            repo_id,
                            "--local-dir", str(local_dir),
                            "--local-dir-use-symlinks", "False"
                        ],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    logger.info(f"✓ Downloaded {local_name}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to download {repo_id}: {e.stderr}")
                    raise
            else:
                logger.info(f"✓ {local_name} already exists")
        
        logger.info("✓ All checkpoints ready")
    
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
        """Generate talking head video using InfiniteTalk CLI."""

        # Ensure initialized
        self._ensure_installed()

        # Validate inputs
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Create temporary directories
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Save input image
            input_image_path = temp_path / "input_face.png"
            image = self._preprocess_image(image)
            image.save(input_image_path)

            # Create input JSON for InfiniteTalk
            input_json_path = temp_path / "input.json"
            input_json = {
                "prompt": "high quality talking head video",
                "cond_video": str(input_image_path),
                "cond_audio": {
                    "person1": str(audio_path)
                },
                "audio_type": "para"
            }

            with open(input_json_path, 'w') as f:
                json.dump(input_json, f)

            # Prepare output path
            output_base = temp_path / "output"

            # Run InfiniteTalk generation
            logger.info(f"Running InfiniteTalk with {num_frames} frames at {fps} FPS...")

            weights_dir = self.base_path / "weights"

            # Find the actual inference script
            inference_script = self.repo_path / "generate_infinitetalk.py"
            if not inference_script.exists():
                raise FileNotFoundError(f"InfiniteTalk script not found at {inference_script}")

            # Build command with correct arguments
            cmd = [
                "python", str(inference_script),
                "--task", "infinitetalk-14B",
                "--size", "infinitetalk-480",
                "--ckpt_dir", str(weights_dir / "Wan2.1-I2V-14B-480P"),
                "--infinitetalk_dir", str(weights_dir / "InfiniteTalk"),
                "--wav2vec_dir", str(weights_dir / "chinese-wav2vec2-base"),
                "--input_json", str(input_json_path),
                "--save_file", str(output_base),
                "--audio_mode", "localfile",
                "--mode", "clip",
                "--frame_num", str(num_frames),
                "--sample_steps", "40",
                "--motion_frame", "9"
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,
                    cwd=str(self.repo_path)
                )

                if result.returncode != 0:
                    logger.error(f"InfiniteTalk failed: {result.stderr}")
                    raise RuntimeError(f"InfiniteTalk generation failed: {result.stderr}")

                logger.info("✓ InfiniteTalk generation complete")

                # Find output video (script adds .mp4 extension)
                output_video = Path(f"{output_base}.mp4")
                if not output_video.exists():
                    # Try alternative patterns
                    possible_outputs = list(temp_path.glob("*.mp4"))
                    if possible_outputs:
                        output_video = possible_outputs[0]
                    else:
                        raise FileNotFoundError("InfiniteTalk did not produce output video")

                # Extract frames from video
                frames = self._extract_frames(output_video, num_frames)

                logger.info(f"✓ Extracted {len(frames)} frames")
                return frames

            except subprocess.TimeoutExpired:
                raise RuntimeError("InfiniteTalk generation timed out (>10 minutes)")
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                raise
    
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
    
    def _extract_frames(self, video_path: Path, num_frames: int) -> List[Image.Image]:
        """Extract frames from generated video using ffmpeg."""
        import cv2
        
        frames = []
        cap = cv2.VideoCapture(str(video_path))
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for i in range(min(num_frames, total_frames)):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
        
        cap.release()
        
        return frames
    
    def to(self, device: str):
        """Move model to device."""
        self.device = device
        return self


def get_infinitetalk_pipeline(device: str = "cuda") -> InfiniteTalkPipeline:
    """Get InfiniteTalk pipeline instance."""
    return InfiniteTalkPipeline(device=device)
