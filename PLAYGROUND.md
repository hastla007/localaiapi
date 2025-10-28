# 🎮 Add Playground to Your Dashboard

Quick update to add interactive model testing to your existing dashboard!

## What's New?

The updated dashboard adds a **Playground** tab where you can:
- 🎨 Test text-to-image models (Flux, SDXL, SD3)
- 🔍 Caption images (LLaVA, BLIP-2)
- 🎬 Animate images into videos (SVD)
- All directly in your browser!

## 📦 What You Need

Just ONE file: `dashboard.html` (updated version)

## ⚡ Installation (30 seconds)

### If You Already Have the Dashboard

1. **Stop Container**
   ```bash
   cd your-ai-api-project
   docker-compose down
   ```

2. **Replace dashboard.html**
   - Delete old `templates/dashboard.html`
   - Copy new `dashboard.html` to `templates/`

3. **Restart Container**
   ```bash
   docker-compose up -d
   ```

4. **Open Dashboard**
   ```
   http://localhost:8000/dashboard
   ```

5. **Click "Playground" tab** - You're done! 🎉

### No Code Changes Needed!

The new dashboard works with your existing `main.py` and API - just replace the HTML file.

## ✨ Features Added

### Text-to-Image Panel
- Model selector (Flux, SDXL, SD3)
- Prompt and negative prompt inputs
- Resolution controls (512-1024)
- Steps and guidance sliders
- Seed input for reproducibility
- Live preview of generated images
- Download button

### Image-to-Text Panel
- Model selector (LLaVA, BLIP-2)
- Drag & drop image upload
- Custom prompt input
- Live preview of uploaded image
- Caption display with timing

### Image-to-Video Panel
- Image upload
- Frame count slider (8-25)
- Steps slider (10-50)
- FPS control (5-30)
- Motion intensity slider
- Video download after generation

## 🎯 Quick Test

After updating:

1. Go to **Playground** tab
2. Enter prompt: "a beautiful sunset"
3. Click **Generate Image**
4. Wait 15-30 seconds
5. Image appears with download button!

## 📊 All Original Features Still Work

✅ Overview tab - Status & models  
✅ Results gallery  
✅ Metrics & charts  
✅ Logs viewer  
✅ Settings page  
✅ All API endpoints  

## 💡 Why Update?

**Before**: Test models via API calls/n8n  
**After**: Test models instantly in browser  

Perfect for:
- Quick prompt testing
- Finding optimal parameters
- Visual feedback
- Non-technical users
- Rapid prototyping

## 🔄 Rollback (If Needed)

Keep your old `dashboard.html` as backup:
```bash
cp templates/dashboard.html templates/dashboard_old.html
```

To rollback:
```bash
cp templates/dashboard_old.html templates/dashboard.html
docker-compose restart
```

## 📚 Documentation

- `PLAYGROUND_GUIDE.md` - Complete usage guide
- Tips, workflows, and best practices
- Example prompts and presets

## 🎨 Example Uses

### Quick Testing
```
1. Try prompt in Playground
2. See result in 20 seconds
3. Adjust parameters
4. Generate again
```

### Finding Best Settings
```
1. Test in Playground with SDXL (fast)
2. Note what works
3. Use same settings in n8n with Flux (quality)
```

### Demos & Presentations
```
1. Open Playground tab
2. Show live image generation
3. Caption the result
4. Animate to video
All in real-time!
```

## 🚀 Pro Tips

1. **Use SDXL for testing** - Fast iteration
2. **Switch to Flux for finals** - Best quality
3. **Monitor VRAM in Overview** - Unload if needed
4. **Save good prompts** - Build your library
5. **Test before n8n** - Verify parameters work

## ⚙️ No Configuration Required

The Playground uses your existing:
- Model cache
- Output directory
- API endpoints
- Settings

Everything just works!

## 🆘 Troubleshooting

### Playground tab not showing?
- Clear browser cache (Ctrl+F5)
- Check file copied to `templates/` folder

### Uploads not working?
- Make sure file is image format (PNG, JPG)
- Check file size < 10MB

### Generation fails?
- Check **Logs** tab for errors
- Try **Unload All Models** first
- Use lower resolution/steps

### Slow generation?
- Normal on first use (model loading)
- Check **Overview** for loaded models
- Subsequent runs are faster

## 📝 Version Info

**Dashboard Version**: 1.1.0  
**New Features**: Interactive Playground  
**Compatibility**: Works with existing API v1.0.0  
**File Changed**: Only `dashboard.html`  

---

## 🎉 You're Ready!

The Playground is a game-changer for:
- ✅ Faster testing
- ✅ Better prompts
- ✅ Visual feedback
- ✅ Easier demos
- ✅ More fun! 🎨

**Total Update Time: 30 seconds**

**Start creating: http://localhost:8000/dashboard** 🚀

---

Questions? Check `PLAYGROUND_GUIDE.md` for complete documentation!
