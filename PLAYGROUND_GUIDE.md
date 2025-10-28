# 🎮 Playground Guide - Test Your AI Models

The Playground tab lets you test all AI models directly in your browser - no code needed!

## 🎨 Text-to-Image Generation

Generate images from text descriptions using three powerful models:

### Available Models

1. **Flux.1-dev** - Best Quality
   - Highest quality outputs
   - Best for detailed, artistic images
   - Slower (30-60s)
   - VRAM: ~12GB

2. **SDXL** - Faster (Recommended for testing)
   - Good quality
   - Fast generation (15-30s)
   - VRAM: ~7GB

3. **Stable Diffusion 3**
   - High quality
   - Moderate speed
   - VRAM: ~10GB

### How to Use

1. **Select Model**: Choose from Flux, SDXL, or SD3
2. **Enter Prompt**: Describe what you want to see
   - Example: "a majestic dragon flying over mountains at sunset, digital art, highly detailed"
3. **Negative Prompt** (optional): What to avoid
   - Example: "blurry, low quality, distorted"
4. **Set Image Size**: 512, 768, or 1024 pixels
5. **Adjust Steps**: 10-50 (higher = better quality but slower)
6. **Guidance Scale**: 1-20 (7.5 is optimal)
7. **Seed** (optional): For reproducible results
8. **Click "Generate Image"**

### Tips for Better Results

✅ **Be Descriptive**: "a red sports car on a mountain road at sunset" beats "car"
✅ **Add Style Tags**: Include "digital art", "photorealistic", "oil painting", etc.
✅ **Use Negative Prompts**: Avoid common issues like "blurry, distorted, watermark"
✅ **Start with SDXL**: Faster for testing prompts
✅ **Use Flux for Finals**: Switch to Flux for best quality final images

### Common Issues

❌ **Generation Too Slow**: Reduce steps to 20-25 or use smaller resolution
❌ **Poor Quality**: Increase steps to 40-50 and use Flux model
❌ **Out of Memory**: Unload other models first or reduce resolution

---

## 🔍 Image-to-Text (Captioning)

Describe images using AI vision models.

### Available Models

1. **LLaVA 1.6** - Detailed Descriptions
   - Very detailed captions
   - Better at understanding context
   - VRAM: ~13GB

2. **BLIP-2** - Faster Captions
   - Quick, accurate captions
   - Good for basic descriptions
   - VRAM: ~6GB

### How to Use

1. **Select Model**: LLaVA for detail, BLIP-2 for speed
2. **Upload Image**: Click "Upload Image" and select file
3. **Customize Prompt** (LLaVA only):
   - "Describe this image in detail"
   - "What objects are in this image?"
   - "What is the mood of this scene?"
4. **Click "Generate Caption"**

### Use Cases

📸 **Content Moderation**: Automatically describe uploaded images
📝 **Alt Text Generation**: Create accessibility descriptions
🔍 **Image Search**: Generate searchable descriptions
📊 **Dataset Labeling**: Auto-caption large image collections

### Example Prompts

- "Describe this image in detail, including colors, objects, and atmosphere"
- "What is the main subject of this image?"
- "List all objects visible in this image"
- "What emotions does this image convey?"

---

## 🎬 Image-to-Video (Animation)

Animate static images using Stable Video Diffusion (SVD).

### What It Does

Takes a still image and creates a short video (2-3 seconds) with motion:
- Camera movements
- Object animations
- Subtle dynamics
- Natural-looking motion

### How to Use

1. **Upload Image**: Choose a high-quality image
   - Will be resized to 1024x576
   - Best results with clear subjects
2. **Set Parameters**:
   - **Frames**: 8-25 (14 recommended)
   - **Steps**: 10-50 (25 recommended)
   - **FPS**: 5-30 frames per second (7 recommended)
   - **Motion Intensity**: 1-255 (127 = medium motion)
3. **Click "Generate Video"**
4. **Wait**: 1-2 minutes for processing
5. **Download**: MP4 video file

### Parameters Explained

**Number of Frames**
- 8 frames = ~1 second video
- 14 frames = ~2 seconds (recommended)
- 25 frames = ~3.5 seconds
- More frames = longer video but slower generation

**Steps**
- 10-15: Fast but lower quality
- 20-25: Balanced (recommended)
- 30-50: Highest quality but slow

**FPS (Frames Per Second)**
- 5-7: Slower, dramatic motion
- 10-15: Natural motion (recommended)
- 20-30: Fast, energetic motion

**Motion Intensity**
- Low (50-100): Subtle, gentle movement
- Medium (127): Balanced motion
- High (150-255): Strong, dynamic movement

### Best Practices

✅ **Start Simple**: Use 14 frames, 25 steps, 7 FPS
✅ **Clear Subject**: Images with distinct subjects work best
✅ **Good Lighting**: Well-lit images produce better results
✅ **Landscape Oriented**: Works best with wide images
✅ **Unload Models**: Free VRAM before generating videos

### What Works Best

👍 **Good for:**
- Portraits (adds subtle movement)
- Landscapes (camera pans, clouds move)
- Objects on plain backgrounds
- Photos with clear focal points

👎 **Challenging:**
- Complex scenes with many objects
- Low-resolution images
- Very dark or overexposed images
- Cluttered compositions

### Common Issues

❌ **Video Too Short**: Increase number of frames
❌ **Too Much Motion**: Lower motion intensity
❌ **Not Enough Motion**: Increase motion intensity
❌ **Generation Failed**: Unload other models, try smaller frame count
❌ **Poor Quality**: Increase steps to 30-40

---

## 💡 Workflow Tips

### Quick Testing Workflow

1. **Generate Image** (SDXL, 512x512, 20 steps)
2. **Review Result**
3. **Refine Prompt** if needed
4. **Generate Final** (Flux, 1024x1024, 30 steps)
5. **Caption Image** (to verify what AI sees)
6. **Animate** (optional, for presentations)

### Save VRAM

Between generations:
1. Go to **Overview** tab
2. Click **"Unload All Models"**
3. Return to Playground
4. Generate with fresh model

### Batch Testing

Test multiple prompts quickly:
1. Use SDXL model
2. Set to 512x512
3. Use 20 steps
4. Generate multiple variations
5. Pick best, regenerate with Flux at high res

### From Playground to n8n

1. **Test in Playground** to find best parameters
2. **Note successful settings**
3. **Copy prompt and parameters**
4. **Use in n8n HTTP Request node** with same settings

---

## 🎯 Example Workflows

### Workflow 1: Product Photos

1. Generate product image (Flux)
   - Prompt: "studio product photo of [product], white background, professional lighting"
2. Caption with BLIP-2
   - For alt-text and SEO
3. Animate with SVD
   - 14 frames, 7 FPS, medium motion
   - For social media posts

### Workflow 2: Story Illustrations

1. Generate scene (Flux)
   - Detailed prompt with style
2. Caption with LLaVA
   - Verify scene accuracy
3. Generate variations
   - Adjust prompt based on caption

### Workflow 3: Social Media Content

1. Generate hero image (SDXL for speed)
2. Test several prompts
3. Pick best, regenerate with Flux
4. Animate for Instagram/TikTok
5. Caption for accessibility

---

## 📊 Performance Guide

### Generation Times (Approximate)

**Text-to-Image**
- SDXL 512x512, 20 steps: 10-15s
- SDXL 1024x1024, 30 steps: 20-30s
- Flux 1024x1024, 30 steps: 30-60s

**Image-to-Text**
- BLIP-2: 5-10s
- LLaVA: 15-30s

**Video**
- 14 frames, 25 steps: 60-120s

### VRAM Usage

Monitor VRAM in **Overview** tab:
- Green (<80%): Safe to generate
- Red (>80%): Unload models first

---

## 🚀 Quick Reference

### Text-to-Image Presets

**Fast Testing**
- Model: SDXL
- Size: 512x512
- Steps: 20
- Guidance: 7.5

**High Quality**
- Model: Flux
- Size: 1024x1024
- Steps: 40
- Guidance: 7.5

**Balanced**
- Model: SDXL
- Size: 768x768
- Steps: 30
- Guidance: 7.5

### Video Presets

**Quick Preview**
- Frames: 8
- Steps: 15
- FPS: 7
- Motion: 100

**High Quality**
- Frames: 14
- Steps: 30
- FPS: 10
- Motion: 127

**Long Animation**
- Frames: 25
- Steps: 25
- FPS: 15
- Motion: 150

---

## 🎨 Prompt Library

### Photography Styles
- "professional product photography, studio lighting"
- "cinematic photography, golden hour, dramatic lighting"
- "macro photography, extreme close-up, shallow depth of field"
- "aerial drone photography, top-down view"

### Art Styles
- "digital art, vibrant colors, detailed"
- "oil painting, impressionist style"
- "watercolor painting, soft colors"
- "pencil sketch, detailed linework"

### Quality Boosters
- "highly detailed, 8k, professional"
- "award-winning photography"
- "trending on artstation"
- "masterpiece, best quality"

### Negative Prompt Templates
- "blurry, low quality, distorted, ugly, bad anatomy"
- "watermark, text, signature, copyright"
- "cartoon, anime" (for photorealistic)
- "photorealistic" (for artistic styles)

---

**Happy Creating! 🎨✨**
