import requests
import json
import base64
from pathlib import Path

# API base URL
API_URL = "http://localhost:8000"

def test_health_check():
    """Test if API is running"""
    print("\n=== Testing Health Check ===")
    response = requests.get(f"{API_URL}/")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200

def test_list_models():
    """Test listing available models"""
    print("\n=== Testing List Models ===")
    response = requests.get(f"{API_URL}/models")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Available models: {list(data['available_models'].keys())}")
    print(f"Loaded models: {data['loaded_models']}")
    return response.status_code == 200

def test_text_to_image():
    """Test text-to-image generation with SDXL (fastest)"""
    print("\n=== Testing Text-to-Image (SDXL) ===")
    
    payload = {
        "prompt": "a beautiful sunset over mountains, digital art",
        "negative_prompt": "blurry, low quality",
        "width": 512,  # Smaller for faster test
        "height": 512,
        "num_inference_steps": 20,  # Fewer steps for speed
        "guidance_scale": 7.5,
        "seed": 42
    }
    
    print("Sending request... (this may take 30-60 seconds on first run)")
    response = requests.post(
        f"{API_URL}/api/generate/sdxl",
        json=payload,
        timeout=300
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Generation time: {data['generation_time']}s")
        print(f"Image saved to: {data['image_path']}")
        print(f"Base64 length: {len(data['image_base64'])} characters")
        print("✅ Image generation successful!")
        return True
    else:
        print(f"❌ Error: {response.text}")
        return False

def test_image_to_text():
    """Test image-to-text captioning"""
    print("\n=== Testing Image-to-Text (BLIP-2) ===")
    
    # Create a simple test image (red square)
    from PIL import Image
    import io
    
    img = Image.new('RGB', (512, 512), color='red')
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    
    payload = {
        "image": f"data:image/png;base64,{img_base64}",
        "prompt": "Describe this image.",
        "max_length": 100
    }
    
    print("Sending request... (this may take 30-60 seconds on first run)")
    response = requests.post(
        f"{API_URL}/api/caption/blip",
        json=payload,
        timeout=300
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Caption: {data['caption']}")
        print(f"Generation time: {data['generation_time']}s")
        print("✅ Image captioning successful!")
        return True
    else:
        print(f"❌ Error: {response.text}")
        return False

def test_unload_models():
    """Test unloading all models"""
    print("\n=== Testing Unload All Models ===")
    response = requests.post(f"{API_URL}/api/unload-all")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
        print("✅ Models unloaded successfully!")
        return True
    return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("AI API Test Suite")
    print("=" * 60)
    
    tests = [
        ("Health Check", test_health_check),
        ("List Models", test_list_models),
        ("Text-to-Image", test_text_to_image),
        ("Image-to-Text", test_image_to_text),
        ("Unload Models", test_unload_models),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except requests.exceptions.ConnectionError:
            print(f"\n❌ Cannot connect to API at {API_URL}")
            print("Make sure the Docker container is running:")
            print("  docker-compose up -d")
            return
        except Exception as e:
            print(f"\n❌ Error in {test_name}: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! API is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the logs above for details.")

if __name__ == "__main__":
    main()
