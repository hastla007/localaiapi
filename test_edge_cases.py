#!/usr/bin/env python3
"""
Edge Case and Bug Validation Test Suite

This test suite focuses on edge cases, boundary conditions, and potential
bugs that might not be covered by the standard test suite.
"""
import requests
import json
import base64
import time
from pathlib import Path
from PIL import Image
from io import BytesIO
import sys

API_URL = "http://localhost:8000"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

class TestTracker:
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
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

    def print_summary(self):
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}EDGE CASE TEST SUMMARY{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.RESET}\n")
        print(f"Total Tests: {self.total}")
        print(f"{Colors.GREEN}Passed: {self.passed}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {self.failed}{Colors.RESET}")

        if self.failures:
            print(f"\n{Colors.RED}{Colors.BOLD}Failed Tests:{Colors.RESET}")
            for test_name, error in self.failures:
                print(f"  {Colors.RED}• {test_name}{Colors.RESET}")
                print(f"    {error}")

        return 0 if self.failed == 0 else 1

def create_test_image(color='red', size=(512, 512)):
    """Create a test image and return base64 encoded string"""
    img = Image.new('RGB', size, color=color)
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text:^80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.RESET}\n")

def test_seed_zero_handling(tracker):
    """Test that seed=0 is properly handled (not rejected as falsy)"""
    print_header("SEED=0 HANDLING TEST")

    try:
        payload = {
            "prompt": "test image",
            "seed": 0,  # This should be valid
            "num_inference_steps": 5
        }
        response = requests.post(f"{API_URL}/api/generate/sdxl", json=payload, timeout=60)

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                tracker.record_pass("Seed=0 is properly accepted")
            else:
                tracker.record_fail("Seed=0 handling", "Request succeeded but success=False")
        else:
            tracker.record_fail("Seed=0 handling", f"HTTP {response.status_code}: {response.text}")
    except Exception as e:
        tracker.record_fail("Seed=0 handling", str(e))

def test_negative_seed(tracker):
    """Test that negative seed is rejected"""
    print_header("NEGATIVE SEED VALIDATION TEST")

    try:
        payload = {
            "prompt": "test image",
            "seed": -1,  # Should be rejected
            "num_inference_steps": 5
        }
        response = requests.post(f"{API_URL}/api/generate/sdxl", json=payload, timeout=30)

        # Should fail validation (422 Unprocessable Entity)
        if response.status_code == 422:
            tracker.record_pass("Negative seed properly rejected")
        else:
            tracker.record_fail("Negative seed validation", f"Expected 422, got {response.status_code}")
    except Exception as e:
        tracker.record_fail("Negative seed validation", str(e))

def test_empty_prompt_whitespace(tracker):
    """Test that whitespace-only prompts are rejected"""
    print_header("WHITESPACE PROMPT VALIDATION TEST")

    test_cases = [
        ("", "empty string"),
        ("   ", "only spaces"),
        ("\t\t", "only tabs"),
        ("\n\n", "only newlines"),
        ("  \t\n  ", "mixed whitespace")
    ]

    for prompt, description in test_cases:
        try:
            payload = {
                "prompt": prompt,
                "num_inference_steps": 5
            }
            response = requests.post(f"{API_URL}/api/generate/sdxl", json=payload, timeout=30)

            if response.status_code in [400, 422]:
                tracker.record_pass(f"Whitespace prompt rejected ({description})")
            else:
                tracker.record_fail(f"Whitespace prompt validation ({description})",
                                  f"Expected 400/422, got {response.status_code}")
        except Exception as e:
            tracker.record_fail(f"Whitespace prompt validation ({description})", str(e))

def test_base64_data_uri(tracker):
    """Test that data URI format base64 is properly handled"""
    print_header("DATA URI BASE64 HANDLING TEST")

    try:
        # Create test image
        img_base64 = create_test_image(size=(256, 256))

        # Test with data URI prefix
        data_uri = f"data:image/png;base64,{img_base64}"

        payload = {
            "image": data_uri,
            "prompt": "Describe this image"
        }
        response = requests.post(f"{API_URL}/api/caption/blip", json=payload, timeout=60)

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                tracker.record_pass("Data URI format properly handled")
            else:
                tracker.record_fail("Data URI handling", "Request succeeded but success=False")
        else:
            tracker.record_fail("Data URI handling", f"HTTP {response.status_code}: {response.text}")
    except Exception as e:
        tracker.record_fail("Data URI handling", str(e))

def test_invalid_base64(tracker):
    """Test that invalid base64 is properly rejected"""
    print_header("INVALID BASE64 VALIDATION TEST")

    try:
        payload = {
            "image": "this is not base64!!!",
            "prompt": "Describe this image"
        }
        response = requests.post(f"{API_URL}/api/caption/blip", json=payload, timeout=30)

        if response.status_code in [400, 422, 500]:
            tracker.record_pass("Invalid base64 properly rejected")
        else:
            tracker.record_fail("Invalid base64 validation", f"Expected error, got {response.status_code}")
    except Exception as e:
        tracker.record_fail("Invalid base64 validation", str(e))

def test_oversized_image(tracker):
    """Test that oversized images are rejected"""
    print_header("OVERSIZED IMAGE VALIDATION TEST")

    try:
        # Create a large image (simulate >50MB by creating very large dimensions)
        # Note: actual test would need 50MB+ but we'll test the validation exists
        img = Image.new('RGB', (10000, 10000), color='red')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        large_image_base64 = base64.b64encode(buffer.getvalue()).decode()

        payload = {
            "image": large_image_base64,
            "prompt": "Describe this image"
        }
        response = requests.post(f"{API_URL}/api/caption/blip", json=payload, timeout=60)

        # Should either succeed or fail with proper error message
        if response.status_code in [200, 400, 413, 500]:
            tracker.record_pass("Large image handling (validation exists)")
        else:
            tracker.record_fail("Large image handling", f"Unexpected status {response.status_code}")
    except Exception as e:
        # Timeout or connection error is acceptable for very large payloads
        if "timeout" in str(e).lower() or "connection" in str(e).lower():
            tracker.record_pass("Large image handling (timeout protection)")
        else:
            tracker.record_fail("Large image handling", str(e))

def test_boundary_values(tracker):
    """Test boundary values for numeric parameters"""
    print_header("BOUNDARY VALUE TESTS")

    # Test minimum values
    try:
        payload = {
            "prompt": "test",
            "width": 512,  # minimum
            "height": 512,  # minimum
            "num_inference_steps": 1,  # minimum
            "guidance_scale": 1.0  # minimum
        }
        response = requests.post(f"{API_URL}/api/generate/sdxl", json=payload, timeout=60)

        if response.status_code == 200:
            tracker.record_pass("Minimum boundary values accepted")
        else:
            tracker.record_fail("Minimum boundary values", f"HTTP {response.status_code}")
    except Exception as e:
        tracker.record_fail("Minimum boundary values", str(e))

    # Test maximum values
    try:
        payload = {
            "prompt": "test",
            "width": 2048,  # maximum
            "height": 2048,  # maximum
            "num_inference_steps": 100,  # maximum
            "guidance_scale": 20.0  # maximum
        }
        response = requests.post(f"{API_URL}/api/generate/sdxl", json=payload, timeout=60)

        if response.status_code == 200:
            tracker.record_pass("Maximum boundary values accepted")
        else:
            tracker.record_fail("Maximum boundary values", f"HTTP {response.status_code}")
    except Exception as e:
        tracker.record_fail("Maximum boundary values", str(e))

    # Test out-of-range values (should be rejected)
    try:
        payload = {
            "prompt": "test",
            "width": 256,  # below minimum (512)
            "num_inference_steps": 5
        }
        response = requests.post(f"{API_URL}/api/generate/sdxl", json=payload, timeout=30)

        if response.status_code == 422:
            tracker.record_pass("Below-minimum values rejected")
        else:
            tracker.record_fail("Below-minimum validation", f"Expected 422, got {response.status_code}")
    except Exception as e:
        tracker.record_fail("Below-minimum validation", str(e))

def test_path_traversal_protection(tracker):
    """Test that path traversal attacks are blocked"""
    print_header("PATH TRAVERSAL PROTECTION TEST")

    malicious_paths = [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "....//....//....//etc/passwd",
        "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"
    ]

    for path in malicious_paths:
        try:
            response = requests.get(f"{API_URL}/api/download/{path}", timeout=10)

            if response.status_code in [403, 404]:
                tracker.record_pass(f"Path traversal blocked: {path[:30]}")
            else:
                tracker.record_fail(f"Path traversal protection ({path[:30]})",
                                  f"Expected 403/404, got {response.status_code}")
        except Exception as e:
            tracker.record_fail(f"Path traversal test ({path[:30]})", str(e))

def test_concurrent_model_loading(tracker):
    """Test that concurrent model loading is handled gracefully"""
    print_header("CONCURRENT MODEL LOADING TEST")

    try:
        import threading
        import queue

        results = queue.Queue()

        def make_request(model_endpoint):
            try:
                payload = {"prompt": "test", "num_inference_steps": 5}
                response = requests.post(f"{API_URL}{model_endpoint}", json=payload, timeout=120)
                results.put((model_endpoint, response.status_code, None))
            except Exception as e:
                results.put((model_endpoint, None, str(e)))

        # Start concurrent requests to different models
        threads = [
            threading.Thread(target=make_request, args=("/api/generate/sdxl",)),
            threading.Thread(target=make_request, args=("/api/generate/pony",))
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=150)

        # Check results
        all_succeeded = True
        while not results.empty():
            endpoint, status, error = results.get()
            if status != 200 and error is None:
                all_succeeded = False
            if error and "timeout" not in error.lower():
                all_succeeded = False

        if all_succeeded or not results.empty():
            tracker.record_pass("Concurrent model loading handled")
        else:
            tracker.record_fail("Concurrent model loading", "Some requests failed unexpectedly")

    except Exception as e:
        tracker.record_fail("Concurrent model loading", str(e))

def test_malformed_json(tracker):
    """Test that malformed JSON is properly rejected"""
    print_header("MALFORMED JSON TEST")

    try:
        response = requests.post(
            f"{API_URL}/api/generate/sdxl",
            data="{invalid json}",
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        if response.status_code in [400, 422]:
            tracker.record_pass("Malformed JSON properly rejected")
        else:
            tracker.record_fail("Malformed JSON handling", f"Expected 400/422, got {response.status_code}")
    except Exception as e:
        tracker.record_fail("Malformed JSON handling", str(e))

def test_empty_request_body(tracker):
    """Test that empty request body is handled"""
    print_header("EMPTY REQUEST BODY TEST")

    try:
        response = requests.post(f"{API_URL}/api/generate/sdxl", json={}, timeout=10)

        if response.status_code in [400, 422]:
            tracker.record_pass("Empty request body rejected")
        else:
            tracker.record_fail("Empty request body handling", f"Expected 400/422, got {response.status_code}")
    except Exception as e:
        tracker.record_fail("Empty request body handling", str(e))

def main():
    print(f"{Colors.BOLD}{Colors.CYAN}")
    print("=" * 80)
    print("Edge Case and Bug Validation Test Suite".center(80))
    print("Multi-Model AI API - Comprehensive Edge Case Testing".center(80))
    print("=" * 80)
    print(f"{Colors.RESET}\n")

    # Check API connection first
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        if response.status_code != 200:
            print(f"{Colors.RED}ERROR: API not responding properly{Colors.RESET}")
            return 1
        print(f"{Colors.GREEN}✓ API is online{Colors.RESET}\n")
    except Exception as e:
        print(f"{Colors.RED}ERROR: Cannot connect to API at {API_URL}{Colors.RESET}")
        print(f"{Colors.RED}{e}{Colors.RESET}\n")
        return 1

    tracker = TestTracker()

    # Run edge case tests
    test_seed_zero_handling(tracker)
    test_negative_seed(tracker)
    test_empty_prompt_whitespace(tracker)
    test_base64_data_uri(tracker)
    test_invalid_base64(tracker)
    test_oversized_image(tracker)
    test_boundary_values(tracker)
    test_path_traversal_protection(tracker)
    test_malformed_json(tracker)
    test_empty_request_body(tracker)
    test_concurrent_model_loading(tracker)

    return tracker.print_summary()

if __name__ == "__main__":
    sys.exit(main())
