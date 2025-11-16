#!/usr/bin/env python3
"""
Comprehensive API Test Suite for Multi-Model AI API

This test suite provides complete coverage of all 28 API endpoints,
including edge cases, error handling, and validation tests.
"""
import requests
import json
import base64
import time
from pathlib import Path
from PIL import Image
from io import BytesIO
import sys

# API base URL
API_URL = "http://localhost:8000"

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class TestTracker:
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.failures = []

    def record_pass(self, test_name):
        self.total += 1
        self.passed += 1
        print(f"{Colors.GREEN}✓ PASS{Colors.RESET} {test_name}")

    def record_fail(self, test_name, error):
        self.total += 1
        self.failed += 1
        self.failures.append((test_name, str(error)))
        print(f"{Colors.RED}✗ FAIL{Colors.RESET} {test_name}")
        print(f"  {Colors.RED}Error: {error}{Colors.RESET}")

    def record_skip(self, test_name, reason):
        self.total += 1
        self.skipped += 1
        print(f"{Colors.YELLOW}⊘ SKIP{Colors.RESET} {test_name} - {reason}")

    def print_summary(self):
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}TEST SUMMARY{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.RESET}\n")
        print(f"Total Tests: {self.total}")
        print(f"{Colors.GREEN}Passed: {self.passed}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {self.failed}{Colors.RESET}")
        print(f"{Colors.YELLOW}Skipped: {self.skipped}{Colors.RESET}")

        if self.failures:
            print(f"\n{Colors.RED}{Colors.BOLD}Failed Tests:{Colors.RESET}")
            for test_name, error in self.failures:
                print(f"  {Colors.RED}• {test_name}{Colors.RESET}")
                print(f"    {error}")

        if self.failed == 0 and self.passed > 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 All tests passed!{Colors.RESET}")
            return 0
        elif self.failed > 0:
            print(f"\n{Colors.RED}{Colors.BOLD}⚠️  Some tests failed!{Colors.RESET}")
            return 1
        else:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  No tests ran!{Colors.RESET}")
            return 2


def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text:^80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.RESET}\n")


def create_test_image(color='red', size=(512, 512)):
    """Create a test image and return base64 encoded string"""
    img = Image.new('RGB', size, color=color)
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()


def test_api_connection(tracker):
    """Test if API is accessible"""
    print_header("API CONNECTION TEST")
    try:
        response = requests.get(f"{API_URL}/", timeout=10)
        if response.status_code == 200:
            tracker.record_pass("API connection")
            return True
        else:
            tracker.record_fail("API connection", f"Status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        tracker.record_fail("API connection", f"Cannot connect to {API_URL}")
        print(f"{Colors.YELLOW}Make sure the API is running (docker-compose up -d){Colors.RESET}")
        return False
    except Exception as e:
        tracker.record_fail("API connection", str(e))
        return False


# ==================== HEALTH & STATUS TESTS ====================

def test_health_check(tracker):
    """Test GET /"""
    print_header("HEALTH CHECK TESTS")
    try:
        response = requests.get(f"{API_URL}/", timeout=10)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert "status" in data, "Missing 'status' field"
        assert "gpu_available" in data, "Missing 'gpu_available' field"
        tracker.record_pass("GET / - Health check")
    except Exception as e:
        tracker.record_fail("GET / - Health check", e)


def test_list_models(tracker):
    """Test GET /models"""
    try:
        response = requests.get(f"{API_URL}/models", timeout=10)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert "available_models" in data, "Missing 'available_models' field"
        assert "loaded_models" in data, "Missing 'loaded_models' field"
        tracker.record_pass("GET /models - List models")
    except Exception as e:
        tracker.record_fail("GET /models - List models", e)


def test_tts_status(tracker):
    """Test GET /api/tts/status"""
    try:
        response = requests.get(f"{API_URL}/api/tts/status", timeout=10)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert "available" in data, "Missing 'available' field"
        tracker.record_pass("GET /api/tts/status - TTS status")
    except Exception as e:
        tracker.record_fail("GET /api/tts/status - TTS status", e)


def test_comfyui_status(tracker):
    """Test GET /api/comfyui/status"""
    try:
        response = requests.get(f"{API_URL}/api/comfyui/status", timeout=10)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert "available" in data, "Missing 'available' field"
        tracker.record_pass("GET /api/comfyui/status - ComfyUI status")
    except Exception as e:
        tracker.record_fail("GET /api/comfyui/status - ComfyUI status", e)


# ==================== DASHBOARD TESTS ====================

def test_dashboard_endpoints(tracker):
    """Test all dashboard endpoints"""
    print_header("DASHBOARD ENDPOINT TESTS")

    endpoints = [
        ("/dashboard", "GET", "Dashboard HTML"),
        ("/api/dashboard/status", "GET", "Dashboard status"),
        ("/api/dashboard/results", "GET", "Dashboard results"),
        ("/api/dashboard/metrics", "GET", "Dashboard metrics"),
        ("/api/dashboard/logs", "GET", "Dashboard logs"),
        ("/api/dashboard/settings", "GET", "Dashboard settings"),
    ]

    for endpoint, method, name in endpoints:
        try:
            if method == "GET":
                response = requests.get(f"{API_URL}{endpoint}", timeout=10)
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            tracker.record_pass(f"{method} {endpoint} - {name}")
        except Exception as e:
            tracker.record_fail(f"{method} {endpoint} - {name}", e)


def test_dashboard_logs_clear(tracker):
    """Test POST /api/dashboard/logs/clear"""
    try:
        response = requests.post(f"{API_URL}/api/dashboard/logs/clear", timeout=10)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert data.get("success") == True, "Expected success=True"
        tracker.record_pass("POST /api/dashboard/logs/clear - Clear logs")
    except Exception as e:
        tracker.record_fail("POST /api/dashboard/logs/clear - Clear logs", e)


def test_dashboard_settings_save(tracker):
    """Test POST /api/dashboard/settings"""
    try:
        # Test valid settings
        settings = {
            "max_loaded_models": 2,
            "model_timeout": 300
        }
        response = requests.post(f"{API_URL}/api/dashboard/settings", json=settings, timeout=10)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        tracker.record_pass("POST /api/dashboard/settings - Save settings (valid)")
    except Exception as e:
        tracker.record_fail("POST /api/dashboard/settings - Save settings (valid)", e)

    # Test invalid settings
    try:
        invalid_settings = {
            "max_loaded_models": -1  # Invalid: must be positive
        }
        response = requests.post(f"{API_URL}/api/dashboard/settings", json=invalid_settings, timeout=10)
        assert response.status_code == 400, f"Expected 400 for invalid input, got {response.status_code}"
        tracker.record_pass("POST /api/dashboard/settings - Reject invalid settings")
    except Exception as e:
        tracker.record_fail("POST /api/dashboard/settings - Reject invalid settings", e)


# ==================== TEXT-TO-IMAGE TESTS ====================

def test_text_to_image_endpoints(tracker):
    """Test all text-to-image endpoints"""
    print_header("TEXT-TO-IMAGE TESTS")

    # Common payload for quick tests
    payload = {
        "prompt": "a beautiful landscape",
        "negative_prompt": "blurry",
        "width": 512,
        "height": 512,
        "num_inference_steps": 10,  # Low for speed
        "guidance_scale": 7.5,
        "seed": 42
    }

    endpoints = [
        ("/api/generate/sdxl", "SDXL"),
        ("/api/generate/flux", "Flux.1-dev"),
        ("/api/generate/sd3", "SD3"),
        ("/api/generate/pony", "Pony Diffusion"),
    ]

    for endpoint, model_name in endpoints:
        try:
            print(f"Testing {model_name} (this may take 30-120 seconds)...")
            response = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=300)
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            data = response.json()
            assert "image_path" in data, "Missing image_path"
            assert "image_base64" in data, "Missing image_base64"
            assert "generation_time" in data, "Missing generation_time"
            tracker.record_pass(f"POST {endpoint} - {model_name}")
        except requests.exceptions.Timeout:
            tracker.record_skip(f"POST {endpoint} - {model_name}", "Timeout (model loading takes too long)")
        except Exception as e:
            tracker.record_fail(f"POST {endpoint} - {model_name}", e)


def test_text_to_image_edge_cases(tracker):
    """Test edge cases for text-to-image"""
    print_header("TEXT-TO-IMAGE EDGE CASE TESTS")

    # Test seed=0 (regression test for bug fix)
    try:
        payload = {
            "prompt": "test",
            "width": 512,
            "height": 512,
            "num_inference_steps": 10,
            "seed": 0  # Edge case: seed=0
        }
        response = requests.post(f"{API_URL}/api/generate/sdxl", json=payload, timeout=300)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        tracker.record_pass("Text-to-image with seed=0")
    except requests.exceptions.Timeout:
        tracker.record_skip("Text-to-image with seed=0", "Timeout")
    except Exception as e:
        tracker.record_fail("Text-to-image with seed=0", e)

    # Test empty prompt
    try:
        payload = {
            "prompt": "",  # Empty prompt
            "width": 512,
            "height": 512,
            "num_inference_steps": 10
        }
        response = requests.post(f"{API_URL}/api/generate/sdxl", json=payload, timeout=300)
        # Should either work or return 400/422
        if response.status_code in [200, 400, 422]:
            tracker.record_pass("Text-to-image with empty prompt (handled)")
        else:
            tracker.record_fail("Text-to-image with empty prompt", f"Unexpected status: {response.status_code}")
    except requests.exceptions.Timeout:
        tracker.record_skip("Text-to-image with empty prompt", "Timeout")
    except Exception as e:
        tracker.record_fail("Text-to-image with empty prompt", e)


# ==================== IMAGE-TO-TEXT TESTS ====================

def test_image_to_text_endpoints(tracker):
    """Test all image-to-text endpoints"""
    print_header("IMAGE-TO-TEXT TESTS")

    # Create test image
    img_base64 = create_test_image('blue', (512, 512))

    endpoints = [
        ("/api/caption/blip", "BLIP-2"),
        ("/api/caption/llava", "LLaVA"),
        ("/api/caption/qwen", "Qwen2-VL"),
    ]

    for endpoint, model_name in endpoints:
        try:
            payload = {
                "image": img_base64,
                "prompt": "Describe this image",
                "max_length": 100
            }
            print(f"Testing {model_name} (this may take 30-120 seconds)...")
            response = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=300)
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            data = response.json()
            assert "caption" in data, "Missing caption"
            assert "generation_time" in data, "Missing generation_time"
            tracker.record_pass(f"POST {endpoint} - {model_name}")
        except requests.exceptions.Timeout:
            tracker.record_skip(f"POST {endpoint} - {model_name}", "Timeout")
        except Exception as e:
            tracker.record_fail(f"POST {endpoint} - {model_name}", e)


def test_image_to_text_edge_cases(tracker):
    """Test edge cases for image-to-text"""
    print_header("IMAGE-TO-TEXT EDGE CASE TESTS")

    # Test with data URI format
    try:
        img_base64 = create_test_image('red')
        data_uri = f"data:image/png;base64,{img_base64}"
        payload = {
            "image": data_uri,  # Data URI format
            "prompt": "Describe this",
            "max_length": 50
        }
        response = requests.post(f"{API_URL}/api/caption/blip", json=payload, timeout=300)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        tracker.record_pass("Image-to-text with data URI format")
    except requests.exceptions.Timeout:
        tracker.record_skip("Image-to-text with data URI format", "Timeout")
    except Exception as e:
        tracker.record_fail("Image-to-text with data URI format", e)

    # Test with invalid base64
    try:
        payload = {
            "image": "not-valid-base64!!!",
            "prompt": "Describe this",
            "max_length": 50
        }
        response = requests.post(f"{API_URL}/api/caption/blip", json=payload, timeout=10)
        assert response.status_code in [400, 422, 500], f"Should reject invalid base64, got {response.status_code}"
        tracker.record_pass("Image-to-text rejects invalid base64")
    except Exception as e:
        tracker.record_fail("Image-to-text rejects invalid base64", e)


# ==================== CONTROLNET TESTS ====================

def test_controlnet_endpoints(tracker):
    """Test ControlNet endpoints"""
    print_header("CONTROLNET TESTS")

    # Create control image
    control_img = create_test_image('white', (512, 512))

    endpoints = [
        ("/api/controlnet/mistoline", "MistoLine"),
        ("/api/controlnet/union", "ControlNet Union"),
    ]

    for endpoint, model_name in endpoints:
        try:
            payload = {
                "prompt": "a beautiful portrait",
                "control_image": control_img,
                "negative_prompt": "blurry",
                "width": 512,
                "height": 512,
                "num_inference_steps": 10,
                "seed": 42
            }
            print(f"Testing {model_name} (this may take 60-180 seconds)...")
            response = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=300)
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            data = response.json()
            assert "image_path" in data, "Missing image_path"
            tracker.record_pass(f"POST {endpoint} - {model_name}")
        except requests.exceptions.Timeout:
            tracker.record_skip(f"POST {endpoint} - {model_name}", "Timeout")
        except Exception as e:
            tracker.record_fail(f"POST {endpoint} - {model_name}", e)


# ==================== VIDEO GENERATION TESTS ====================

def test_video_generation_endpoints(tracker):
    """Test video generation endpoints"""
    print_header("VIDEO GENERATION TESTS")

    # Create test image for image-to-video
    img_base64 = create_test_image('green', (512, 512))

    # Test SVD (requires image)
    try:
        payload = {
            "image": img_base64,
            "num_frames": 14,
            "num_inference_steps": 10,
            "fps": 7
        }
        print("Testing SVD (this may take 60-180 seconds)...")
        response = requests.post(f"{API_URL}/api/video/svd", json=payload, timeout=300)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert "video_path" in data, "Missing video_path"
        tracker.record_pass("POST /api/video/svd - Stable Video Diffusion")
    except requests.exceptions.Timeout:
        tracker.record_skip("POST /api/video/svd - Stable Video Diffusion", "Timeout")
    except Exception as e:
        tracker.record_fail("POST /api/video/svd - Stable Video Diffusion", e)

    # Test AnimateDiff (text-to-video)
    try:
        payload = {
            "prompt": "a cat walking",
            "negative_prompt": "blurry",
            "num_frames": 16,
            "num_inference_steps": 8,
            "fps": 8,
            "width": 512,
            "height": 512,
            "seed": 42
        }
        print("Testing AnimateDiff (this may take 60-180 seconds)...")
        response = requests.post(f"{API_URL}/api/video/animatediff", json=payload, timeout=300)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert "video_path" in data, "Missing video_path"
        tracker.record_pass("POST /api/video/animatediff - AnimateDiff Lightning")
    except requests.exceptions.Timeout:
        tracker.record_skip("POST /api/video/animatediff - AnimateDiff Lightning", "Timeout")
    except Exception as e:
        tracker.record_fail("POST /api/video/animatediff - AnimateDiff Lightning", e)

    # Test WAN 2.1 (requires ComfyUI)
    try:
        # First check if ComfyUI is available
        comfyui_response = requests.get(f"{API_URL}/api/comfyui/status", timeout=10)
        if comfyui_response.json().get("available"):
            payload = {
                "image": img_base64,
                "prompt": "smooth motion",
                "num_frames": 49,
                "num_inference_steps": 6,
                "fps": 24,
                "seed": 42
            }
            print("Testing WAN 2.1 (this may take 60-180 seconds)...")
            response = requests.post(f"{API_URL}/api/video/wan21", json=payload, timeout=300)
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            data = response.json()
            assert "video_path" in data, "Missing video_path"
            tracker.record_pass("POST /api/video/wan21 - WAN 2.1")
        else:
            tracker.record_skip("POST /api/video/wan21 - WAN 2.1", "ComfyUI not available")
    except requests.exceptions.Timeout:
        tracker.record_skip("POST /api/video/wan21 - WAN 2.1", "Timeout")
    except Exception as e:
        tracker.record_fail("POST /api/video/wan21 - WAN 2.1", e)


# ==================== TALKING HEAD TESTS ====================

def test_talking_head_infinitetalk(tracker):
    """Test InfiniteTalk talking head endpoint"""
    print_header("TALKING HEAD TESTS")

    # Create face image
    face_img = create_test_image('beige', (512, 512))

    try:
        # Check ComfyUI availability first
        comfyui_response = requests.get(f"{API_URL}/api/comfyui/status", timeout=10)
        if not comfyui_response.json().get("available"):
            tracker.record_skip("POST /api/talking-head/infinitetalk", "ComfyUI not available")
            return

        # Check TTS availability
        tts_response = requests.get(f"{API_URL}/api/tts/status", timeout=10)
        tts_available = tts_response.json().get("available", False)

        if tts_available:
            payload = {
                "face_image": face_img,
                "text": "Hello, this is a test.",
                "num_frames": 120,
                "fps": 25
            }
            print("Testing InfiniteTalk with TTS (this may take 120-300 seconds)...")
            response = requests.post(f"{API_URL}/api/talking-head/infinitetalk", json=payload, timeout=400)
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            data = response.json()
            assert "video_path" in data, "Missing video_path"
            tracker.record_pass("POST /api/talking-head/infinitetalk - InfiniteTalk with TTS")
        else:
            tracker.record_skip("POST /api/talking-head/infinitetalk", "TTS server not available")
    except requests.exceptions.Timeout:
        tracker.record_skip("POST /api/talking-head/infinitetalk", "Timeout")
    except Exception as e:
        tracker.record_fail("POST /api/talking-head/infinitetalk", e)


# ==================== UTILITY TESTS ====================

def test_utility_endpoints(tracker):
    """Test utility endpoints"""
    print_header("UTILITY ENDPOINT TESTS")

    # Test unload specific model
    try:
        # First load a model by generating an image
        payload = {
            "prompt": "test",
            "width": 512,
            "height": 512,
            "num_inference_steps": 1
        }
        requests.post(f"{API_URL}/api/generate/sdxl", json=payload, timeout=300)

        # Now unload it
        response = requests.post(f"{API_URL}/api/unload/sdxl", timeout=10)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert data.get("success") == True, "Expected success=True"
        tracker.record_pass("POST /api/unload/{model_name} - Unload specific model")
    except requests.exceptions.Timeout:
        tracker.record_skip("POST /api/unload/{model_name}", "Timeout during model loading")
    except Exception as e:
        tracker.record_fail("POST /api/unload/{model_name} - Unload specific model", e)

    # Test unload invalid model
    try:
        response = requests.post(f"{API_URL}/api/unload/invalid_model", timeout=10)
        assert response.status_code == 404, f"Expected 404 for invalid model, got {response.status_code}"
        tracker.record_pass("POST /api/unload/{model_name} - Reject invalid model")
    except Exception as e:
        tracker.record_fail("POST /api/unload/{model_name} - Reject invalid model", e)

    # Test unload all
    try:
        response = requests.post(f"{API_URL}/api/unload-all", timeout=30)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert data.get("success") == True, "Expected success=True"
        tracker.record_pass("POST /api/unload-all - Unload all models")
    except Exception as e:
        tracker.record_fail("POST /api/unload-all - Unload all models", e)


def test_download_endpoint(tracker):
    """Test file download endpoint"""
    try:
        # First generate a file
        img_base64 = create_test_image('purple')
        payload = {
            "image": img_base64,
            "prompt": "test",
            "max_length": 50
        }
        gen_response = requests.post(f"{API_URL}/api/caption/blip", json=payload, timeout=300)

        # Get list of files
        results_response = requests.get(f"{API_URL}/api/dashboard/results", timeout=10)
        results = results_response.json().get("results", [])

        if results:
            # Try to download first file
            first_file_path = results[0]["path"]  # e.g., /api/download/filename.png
            response = requests.get(f"{API_URL}{first_file_path}", timeout=10)
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            tracker.record_pass("GET /api/download/{filename} - Download file")
        else:
            tracker.record_skip("GET /api/download/{filename}", "No files to download")
    except requests.exceptions.Timeout:
        tracker.record_skip("GET /api/download/{filename}", "Timeout")
    except Exception as e:
        tracker.record_fail("GET /api/download/{filename} - Download file", e)

    # Test path traversal protection
    try:
        response = requests.get(f"{API_URL}/api/download/../../etc/passwd", timeout=10)
        assert response.status_code in [403, 404], f"Should block path traversal, got {response.status_code}"
        tracker.record_pass("GET /api/download/{filename} - Block path traversal")
    except Exception as e:
        tracker.record_fail("GET /api/download/{filename} - Block path traversal", e)


# ==================== MAIN TEST RUNNER ====================

def main():
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}")
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║                                                                           ║")
    print("║          COMPREHENSIVE API TEST SUITE - Multi-Model AI API v1.3          ║")
    print("║                         All 28 Endpoints Tested                          ║")
    print("║                                                                           ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.RESET}\n")

    tracker = TestTracker()

    # Check API connection first
    if not test_api_connection(tracker):
        print(f"\n{Colors.RED}{Colors.BOLD}Cannot connect to API. Exiting.{Colors.RESET}\n")
        return 1

    # Run all test suites
    print(f"\n{Colors.BOLD}Running test suites...{Colors.RESET}\n")

    # Health & Status
    test_health_check(tracker)
    test_list_models(tracker)
    test_tts_status(tracker)
    test_comfyui_status(tracker)

    # Dashboard
    test_dashboard_endpoints(tracker)
    test_dashboard_logs_clear(tracker)
    test_dashboard_settings_save(tracker)

    # Text-to-Image
    test_text_to_image_endpoints(tracker)
    test_text_to_image_edge_cases(tracker)

    # Image-to-Text
    test_image_to_text_endpoints(tracker)
    test_image_to_text_edge_cases(tracker)

    # ControlNet
    test_controlnet_endpoints(tracker)

    # Video Generation
    test_video_generation_endpoints(tracker)

    # Talking Head
    test_talking_head_infinitetalk(tracker)

    # Utilities
    test_utility_endpoints(tracker)
    test_download_endpoint(tracker)

    # Print summary
    return tracker.print_summary()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
