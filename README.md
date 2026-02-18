# 🎨 Manga 3D Color Pipeline

Transform B&W manga/comics into **colorized 3D parallax experiences** with one command.

AI-powered colorization + depth map generation + interactive WebGL reader + video export.

## What It Does

1. **Takes** a black & white manga page, folder of pages, or full PDF
2. **Colorizes** it using AI (manga-colorization-v2)
3. **Generates depth maps** using Apple Depth Pro
4. **Produces** colorized PDFs, interactive 3D readers, comparison PDFs, and parallax videos

## Quick Start

```bash
conda activate manga3d

# Full pipeline — PDF in, everything out
python manga_cli.py manga.pdf

# Fast preview
python manga_cli.py manga.pdf --preset draft

# Single image
python manga_cli.py page.jpg

# Folder of images
python manga_cli.py pages/

# Only colorize (skip depth/3D)
python manga_cli.py manga.pdf --steps color

# Add side-by-side comparison PDF (B&W vs Color)
python manga_cli.py manga.pdf --compare

# Process specific pages
python manga_cli.py manga.pdf --pages 1-5

# Generate parallax animation video
python manga_cli.py page.jpg --video
python manga_cli.py manga.pdf --video --video-motion figure8

# Generate shareable GIF
python manga_cli.py page.jpg --video-gif --video-duration 3
```

## Presets

| Preset | Color Size | DPI | Speed | Quality |
|--------|-----------|-----|-------|---------|
| `draft` | 384px | 150 | ~5s/page | Preview |
| `normal` | 576px | 200 | ~20s/page | Balanced (default) |
| `high` | 768px | 300 | ~40s/page | Maximum |

## Outputs

```
output/<name>/
├── <name>_color.pdf          # Colorized PDF
├── <name>_compare.pdf        # Side-by-side B&W vs Color (--compare)
├── <name>_reader.html        # Interactive 3D Reader (WebGL)
├── <name>_parallax.mp4       # Parallax animation video (--video)
├── <name>_parallax.gif       # Parallax animation GIF (--video-gif)
├── page_001_color.png        # Per-page colorized images
├── page_001_depth.png        # Per-page depth maps
├── thumbnails/               # Optimized thumbs for reader
└── _pages/                   # Extracted PDF pages (if PDF input)
```

## Parallax Video Export

Generate smooth parallax animation videos from colorized manga + depth maps. Perfect for sharing on social media.

```bash
# Standalone usage
python export_video.py color.png depth.png -o output.mp4

# Different motion patterns
python export_video.py color.png depth.png --motion figure8
python export_video.py color.png depth.png --motion horizontal
python export_video.py color.png depth.png --motion breathe

# GIF output (smaller, loops automatically)
python export_video.py color.png depth.png --gif -o preview.gif

# Process all pairs in a directory
python export_video.py output/manga_name/ --all
```

**Motion patterns:**
- `circle` — Smooth circular camera orbit (default)
- `figure8` — Figure-8 pattern, more dynamic
- `horizontal` — Side-to-side only
- `breathe` — Gentle pulsing zoom effect

## 3D Reader Features

- **WebGL parallax** with mouse, touch, and gyroscope control
- **5 view modes**: Parallax, Anaglyph 3D, Wiggle 3D, Depth Map, Flat
- **Multi-page navigation** with thumbnails, arrows, keyboard, swipe
- **Reading direction**: RTL (manga) and LTR (western comics)
- **Self-contained**: single HTML file, share anywhere
- **Mobile-ready**: touch, swipe, gyroscope support

## Web UI

A browser-based interface for the full pipeline:

```bash
python web_ui.py
# Open http://localhost:5050
```

Features: drag & drop upload, preset selection, real-time progress, output downloads, past runs gallery.

## Architecture

```
B&W Input → [Extract PDF pages] → [AI Colorization] → [Depth Pro] → [Outputs]
                                        ↓                   ↓
                                   Color pages          Depth maps
                                        ↓                   ↓
                                   Color PDF          3D Reader HTML
                                   Compare PDF        Parallax Video
```

**Batched GPU pipeline** (v2): processes all pages through colorization first, then all through depth, minimizing model switching overhead. CPU-bound work (thumbnails, PDFs) runs in parallel via thread pool.

## Use Cases

- **Manga collectors**: Colorize your favorite B&W manga volumes automatically
- **Comic archivists**: Add depth and color to classic black & white comics
- **Content creators**: Generate eye-catching 3D parallax videos for social media
- **Artists**: Visualize depth in your illustrations
- **Educators**: Create engaging visual materials from public domain comics

## Requirements

- Python 3.10+ with conda
- Apple Silicon Mac (MPS acceleration) or CUDA GPU
- ffmpeg (for video export)
- ~4GB disk for models (colorization + Depth Pro)

## Installation

```bash
# Create conda environment
conda create -n manga3d python=3.10
conda activate manga3d

# Install dependencies
pip install torch torchvision pillow numpy matplotlib flask PyMuPDF

# Clone with submodules
git clone --recursive https://github.com/jrubiosainz/manga-3d-color.git
cd manga-3d-color

# Download Depth Pro model
# Place depth_pro.pt in checkpoints/
```

## License

MIT
