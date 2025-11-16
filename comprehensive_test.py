#!/usr/bin/env python3
"""Comprehensive API Test Suite for Multi-Model AI API

This script tests all API endpoints to ensure they work correctly after bug fixes.
"""
import requests
import json
import base64
import time
from pathlib import Path
from PIL import Image
from io import BytesIO

# API base URL
API_URL = "http://localhost:8000"

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text:^70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.RESET}\n")


def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")


def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")


def print_info(text):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ {text}{Colors.RESET}")


def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")


def create_test_image(color='red', size=(512, 512)):
    """Create a test image for testing"""
    img = Image.new('RGB', size, color=color)
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()


def test_health_check():
    """Test basic health check endpoint"""
    print_header("HEALTH CHECK")
    try:
        response = requests.get(f"{API_URL}/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print_success("API is online")
            print_info(f"GPU Available: {data.get('gpu_available', 'N/A')}")
            print_info(f"GPU Name: {data.get('gpu_name', 'N/A')}")
            print_info(f"Loaded Models: {', '.join(data.get('loaded_models', [])) or 'None'}")
            return True
        else:
            print_error(f"Health check failed with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_error(f"Cannot connect to API at {API_URL}")
        print_warning("Make sure the API is running (docker-compose up -d)")
        return False
    except Exception as e:
        print_error(f"Health check error: {str(e)}")
        return False


def test_list_models():
    """Test listing available models"""
    print_header("LIST MODELS")
    try:
        response = requests.get(f"{API_URL}/models", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print_success("Models endpoint working")
            print_info(f"Available models: {len(data.get('available_models', {}))}")
            print_info(f"Loaded models: {', '.join(data.get('loaded_models', [])) or 'None'}")
            return True
        else:
            print_error(f"List models failed with status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"List models error: {str(e)}")
        return False


def test_seed_handling():
    """Test that seed=0 works correctly (bug fix validation)"""
    print_header("SEED HANDLING TEST (Bug Fix Validation)")
    try:
        payload = {
            "prompt": "test image",
            "width": 512,
            "height": 512,
            "num_inference_steps": 10,
            "seed": 0  # This was broken before the fix
        }

        print_info("Testing SDXL with seed=0 (this validates the bug fix)")
        response = requests.post(
            f"{API_URL}/api/generate/sdxl",
            json=payload,
            timeout=300
        )

        if response.status_code == 200:
            data = response.json()
            print_success(f"SDXL with seed=0 works correctly!")
            print_info(f"Generation time: {data.get('generation_time', 'N/A')}s")
            return True
        else:
            print_error(f"SDXL seed test failed: {response.status_code}")
            print_error(f"Response: {response.text}")
            return False
    except Exception as e:
        print_error(f"Seed handling test error: {str(e)}")
        return False


def test_base64_decoding():
    """Test base64 image decoding with various formats"""
    print_header("BASE64 DECODING TEST (Bug Fix Validation)")
    try:
        # Test 1: Data URI format
        img_base64 = create_test_image('blue')
        data_uri = f"data:image/png;base64,{img_base64}"

        payload = {
            "image": data_uri,
            "prompt": "Describe this image",
            "max_length": 50
        }

        print_info("Testing BLIP with data URI format")
        response = requests.post(
            f"{API_URL}/api/caption/blip",
            json=payload,
            timeout=300
        )

        if response.status_code == 200:
            print_success("Data URI format works correctly!")
            data = response.json()
            print_info(f"Caption: {data.get('caption', 'N/A')[:50]}...")
        else:
            print_error(f"Data URI test failed: {response.status_code}")
            return False

        # Test 2: Plain base64
        payload2 = {
            "image": img_base64,
            "prompt": "Describe this image",
            "max_length": 50
        }

        print_info("Testing BLIP with plain base64")
        response2 = requests.post(
            f"{API_URL}/api/caption/blip",
            json=payload2,
            timeout=300
        )

        if response2.status_code == 200:
            print_success("Plain base64 format works correctly!")
            return True
        else:
            print_error(f"Plain base64 test failed: {response2.status_code}")
            return False

    except Exception as e:
        print_error(f"Base64 decoding test error: {str(e)}")
        return False


def test_tts_status():
    """Test TTS server status"""
    print_header("TTS STATUS CHECK")
    try:
        response = requests.get(f"{API_URL}/api/tts/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('available'):
                print_success("TTS server is available")
                print_info(f"URL: {data.get('url', 'N/A')}")
            else:
                print_warning("TTS server is not available")
                print_info(f"Error: {data.get('error', 'N/A')}")
            return True
        else:
            print_error(f"TTS status check failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"TTS status error: {str(e)}")
        return False


def test_comfyui_status():
    """Test ComfyUI status"""
    print_header("COMFYUI STATUS CHECK")
    try:
        response = requests.get(f"{API_URL}/api/comfyui/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('available'):
                print_success("ComfyUI is available")
                print_info(f"URL: {data.get('url', 'N/A')}")
            else:
                print_warning("ComfyUI is not available")
                print_info(f"Error: {data.get('error', 'N/A')}")
            return True
        else:
            print_error(f"ComfyUI status check failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"ComfyUI status error: {str(e)}")
        return False


def test_dashboard_endpoints():
    """Test dashboard-related endpoints"""
    print_header("DASHBOARD ENDPOINTS")
    endpoints = [
        ("/api/dashboard/status", "Dashboard Status"),
        ("/api/dashboard/metrics", "Dashboard Metrics"),
        ("/api/dashboard/logs", "Dashboard Logs"),
        ("/api/dashboard/results", "Dashboard Results"),
        ("/api/dashboard/settings", "Dashboard Settings"),
    ]

    all_passed = True
    for endpoint, name in endpoints:
        try:
            response = requests.get(f"{API_URL}{endpoint}", timeout=10)
            if response.status_code == 200:
                print_success(f"{name}: OK")
            else:
                print_error(f"{name}: Failed ({response.status_code})")
                all_passed = False
        except Exception as e:
            print_error(f"{name}: Error - {str(e)}")
            all_passed = False

    return all_passed


def test_unload_endpoints():
    """Test model unload endpoints"""
    print_header("MODEL UNLOAD TEST")
    try:
        response = requests.post(f"{API_URL}/api/unload-all", timeout=30)
        if response.status_code == 200:
            data = response.json()
            print_success("Unload all models: OK")
            print_info(f"Loaded models after unload: {', '.join(data.get('loaded_models', [])) or 'None'}")
            return True
        else:
            print_error(f"Unload failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Unload error: {str(e)}")
        return False


def main():
    """Run all tests"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║     COMPREHENSIVE API TEST SUITE - Multi-Model AI API v1.3       ║")
    print("║                     Bug Fix Validation Tests                      ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.RESET}\n")

    tests = [
        ("Health Check", test_health_check, True),
        ("List Models", test_list_models, True),
        ("Dashboard Endpoints", test_dashboard_endpoints, True),
        ("TTS Status", test_tts_status, False),  # Optional
        ("ComfyUI Status", test_comfyui_status, False),  # Optional
        ("Seed Handling (Bug Fix)", test_seed_handling, True),
        ("Base64 Decoding (Bug Fix)", test_base64_decoding, True),
        ("Model Unload", test_unload_endpoints, True),
    ]

    results = []
    for test_name, test_func, is_critical in tests:
        try:
            print_info(f"Running: {test_name}")
            result = test_func()
            results.append((test_name, result, is_critical))
            time.sleep(1)  # Small delay between tests
        except Exception as e:
            print_error(f"Test '{test_name}' crashed: {str(e)}")
            results.append((test_name, False, is_critical))

    # Summary
    print_header("TEST RESULTS SUMMARY")

    critical_passed = sum(1 for _, result, critical in results if result and critical)
    critical_total = sum(1 for _, _, critical in results if critical)
    optional_passed = sum(1 for _, result, critical in results if result and not critical)
    optional_total = sum(1 for _, _, critical in results if not critical)

    for test_name, result, is_critical in results:
        status = f"{Colors.GREEN}✓ PASS{Colors.RESET}" if result else f"{Colors.RED}✗ FAIL{Colors.RESET}"
        critical_tag = f"{Colors.BOLD}(Critical){Colors.RESET}" if is_critical else "(Optional)"
        print(f"{status} {test_name} {critical_tag}")

    print(f"\n{Colors.BOLD}Critical Tests: {critical_passed}/{critical_total} passed{Colors.RESET}")
    print(f"{Colors.BOLD}Optional Tests: {optional_passed}/{optional_total} passed{Colors.RESET}")

    if critical_passed == critical_total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 All critical tests passed! API is working correctly.{Colors.RESET}\n")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}⚠️  Some critical tests failed. Check the logs above.{Colors.RESET}\n")
        return 1


if __name__ == "__main__":
    exit(main())
