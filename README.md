# 🎨 Manga 3D Color

Transform black & white manga pages into **colorized 3D parallax** experiences.

![Pipeline](docs/pipeline.png)

## What it does

1. **Colorize** — AI-powered colorization of B&W manga pages using [manga-colorization-v2](https://github.com/qweasdd/manga-colorization-v2)
2. **Depth Map** — Generate depth maps using [Apple Depth Pro](https://github.com/apple/ml-depth-pro)
3. **3D Parallax** — WebGL reader with real-time parallax, multiple view modes, and page navigation

## Demo

https://github.com/user-attachments/assets/placeholder

## Quick Start

### Prerequisites

- Python 3.9+ with conda/mamba
- Node.js 18+
- macOS (Apple Silicon recommended for MPS acceleration) or Linux with CUDA

### 1. Clone & setup

```bash
git clone https://github.com/jrubiosainz/manga-3d-color.git
cd manga-3d-color
```

### 2. Install Python dependencies

```bash
# Create conda environment
conda create -n manga3d python=3.11 -y
conda activate manga3d

# Install PyTorch (macOS)
pip install torch torchvision

# Install dependencies
pip install -r requirements.txt

# Clone and install colorizer
git clone https://github.com/qweasdd/manga-colorization-v2.git
cd manga-colorization-v2
# Download model weights (see their README)
cd ..

# Clone and install depth estimation
git clone https://github.com/apple/ml-depth-pro.git
cd ml-depth-pro
pip install -e .
cd ..
```

### 3. Process manga pages

```bash
# Single image
python pipeline/manga_pipeline.py image.jpg

# Multiple images
python pipeline/manga_pipeline.py image1.jpg image2.jpg image3.jpg

# Folder of images
python pipeline/manga_pipeline.py ./my-manga-pages/

# PDF
python pipeline/manga_pipeline.py manga.pdf

# Only colorize (skip depth)
python pipeline/manga_pipeline.py image.jpg --steps color

# Only depth map
python pipeline/manga_pipeline.py image.jpg --steps depth
```

Output goes to `output/<basename>/` with:
- `*_color.png` — Colorized version
- `*_depth.png` — Depth map
- `*_3d.png` — 3D composite
- `*_3d_comparison.png` — Side-by-side comparison

### 4. Launch the 3D Reader

```bash
cd reader
npm install
npm start
```

Open `http://localhost:3002` in your browser.

**Load processed pages:**
```
http://localhost:3002?folder=<output-folder-name>
http://localhost:3002?folders=folder1,folder2,folder3
```

Or drag & drop your `*_color.png` + `*_depth.png` files directly into the reader.

## 3D Reader Features

- **WebGL parallax engine** — Real-time depth-based parallax effect
- **5 view modes** — Parallax, Layers, Depth map, Color only, Side-by-side
- **Mouse/touch tracking** — Move cursor to shift perspective
- **Auto-move mode** — Automatic gentle parallax animation
- **Multi-page navigation** — Bottom thumbnail strip + left original panel
- **Adjustable parameters** — Focus plane, layer count, parallax intensity
- **Keyboard shortcuts:**
  - `←` `→` — Previous/next page
  - `Space` — Toggle auto-move
  - `F` — Fullscreen
- **Export** — Save current 3D view as PNG
- **Drag & drop** — Load local images without server

## Architecture

```
manga-3d-color/
├── pipeline/
│   └── manga_pipeline.py    # Core processing pipeline
├── reader/
│   ├── server.js            # Express server (serves images + API)
│   ├── public/
│   │   └── index.html       # WebGL 3D reader (single-file SPA)
│   └── package.json
├── output/                   # Generated output (git-ignored)
├── requirements.txt
└── README.md
```

### Pipeline Flow

```
B&W Manga Page
    │
    ├──► manga-colorization-v2 ──► Color Image (*_color.png)
    │
    └──► Apple Depth Pro ──► Depth Map (*_depth.png)
                                    │
                                    ▼
                          WebGL Parallax Reader
                          (color + depth = 3D!)
```

## Configuration

### Pipeline

| Flag | Description | Default |
|------|-------------|---------|
| `--steps` | `all`, `color`, `depth` | `all` |
| `--output` | Output directory | `./output` |
| `--device` | `mps`, `cuda`, `cpu` | Auto-detect |

### Reader Server

| Env | Description | Default |
|-----|-------------|---------|
| `PORT` | Server port | `3002` |
| `OUTPUT_DIR` | Path to output folder | `../output` |

## Performance

On Apple M1/M2/M3 (MPS):
- Colorization: ~3s per page
- Depth estimation: ~14s per page
- Total: ~17s per page

## Integration

### As a standalone tool
Works out of the box — process images and open the reader.

### With Electron apps
The reader is a single HTML file with no framework dependencies. Embed it in any Electron/Tauri app:
```javascript
// Point to the reader
mainWindow.loadURL('http://localhost:3002?folders=my-manga');
```

### Programmatic API
```python
from pipeline.manga_pipeline import process_image

# Returns paths to generated files
result = process_image('input.jpg', output_dir='./output')
print(result)  # {'color': '...', 'depth': '...', '3d': '...'}
```

## Credits

- [manga-colorization-v2](https://github.com/qweasdd/manga-colorization-v2) — AI manga colorization
- [Apple Depth Pro](https://github.com/apple/ml-depth-pro) — Monocular depth estimation
- Built with ❤️ by [Caelum](https://github.com/jrubiosainz)

## License

MIT
