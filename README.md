# 🎨 Manga 3D Color

Transform black & white manga pages into **colorized 3D parallax** experiences.

![Pipeline: B&W → Color → Depth](docs/pipeline.png)

## What it does

1. **Colorize** — AI-powered colorization of B&W manga pages using [manga-colorization-v2](https://github.com/qweasdd/manga-colorization-v2)
2. **Depth Map** — Generate depth maps using [Apple Depth Pro](https://github.com/apple/ml-depth-pro)
3. **3D Parallax** — WebGL reader with real-time parallax, multiple view modes, and page navigation
4. **Web Transform** — Upload B&W manga directly in the reader UI, transform and view in one workflow

## Examples

<table>
<tr>
<td align="center"><b>One Piece (Luffy)</b></td>
<td align="center"><b>Vagabond</b></td>
</tr>
<tr>
<td><img src="docs/luffy_comparison.png" width="400"></td>
<td><img src="docs/vagabond_comparison.png" width="400"></td>
</tr>
<tr>
<td align="center"><b>Colorized</b></td>
<td align="center"><b>Depth Map</b></td>
</tr>
<tr>
<td><img src="docs/luffy_color.png" width="400"></td>
<td><img src="docs/luffy_depth.png" width="400"></td>
</tr>
<tr>
<td><img src="docs/vagabond_color.png" width="400"></td>
<td><img src="docs/vagabond_depth.png" width="400"></td>
</tr>
</table>

## Quick Start

### Prerequisites

- Python 3.9+ with conda/mamba
- Node.js 18+
- **macOS** (Apple Silicon — MPS), **Linux/Windows** (NVIDIA GPU — CUDA), or CPU fallback

### 1. Clone & setup

```bash
git clone https://github.com/jrubiosainz/manga-3d-color.git
cd manga-3d-color
```

### 2. Install Python dependencies

**macOS:**
```bash
./setup.sh
```

**Windows:**
```batch
setup.bat
```

**Manual setup:**
```bash
# Create conda environment
conda create -n manga3d python=3.11 -y
conda activate manga3d

# Install PyTorch
# macOS (MPS):
pip install torch torchvision
# Windows/Linux (CUDA):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

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

# Folder of images
python pipeline/manga_pipeline.py ./my-manga-pages/

# PDF
python pipeline/manga_pipeline.py manga.pdf

# Only colorize (skip depth)
python pipeline/manga_pipeline.py image.jpg --steps color

# Force specific device
python pipeline/manga_pipeline.py image.jpg --device cuda
python pipeline/manga_pipeline.py image.jpg --device cpu
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

Open `http://localhost:3002` — you can:

- **Load existing results** via drag & drop or URL params: `?folder=luffy`
- **Transform new manga** — click 🎨 Transform, upload B&W images, and process them directly in the browser with real-time progress

## 3D Reader Features

- **WebGL parallax engine** — Real-time depth-based parallax effect
- **5 view modes** — Parallax, Layers, Depth map, Color only, Side-by-side
- **Integrated transform** — Upload B&W manga → colorize + depth → auto-load in 3D viewer
- **Real-time progress** — Watch the pipeline process your pages with live log output
- **Mouse/touch/gyroscope tracking** — Shift perspective naturally
- **Auto-move mode** — Gentle automatic parallax animation
- **Multi-page navigation** — Thumbnail strip + keyboard shortcuts
- **Adjustable parameters** — Focus plane, layer count, parallax intensity
- **Export** — Save current 3D view as PNG
- **Drag & drop** — Load local images without server

**Keyboard shortcuts:** `←` `→` navigate pages · `Space` toggle auto-move · `F` fullscreen

## Device Support

The pipeline auto-detects the best available compute device:

| Platform | Device | Flag | Performance |
|----------|--------|------|-------------|
| macOS (Apple Silicon) | MPS | `--device mps` | ~17s/page |
| Windows/Linux (NVIDIA) | CUDA | `--device cuda` | ~10-20s/page |
| Any | CPU | `--device cpu` | ~60-120s/page |

Auto-detection order: **MPS → CUDA → CPU**

## Architecture

```
manga-3d-color/
├── pipeline/
│   └── manga_pipeline.py    # Core processing pipeline
├── reader/
│   ├── server.js            # Express server + transform API
│   └── public/
│       └── index.html       # WebGL 3D reader + transform UI
├── output/                   # Generated output (git-ignored)
├── setup.sh                  # macOS/Linux setup
├── setup.bat                 # Windows setup
├── requirements.txt
└── README.md
```

### Pipeline Flow

```
B&W Manga Page
    │
    ├──► manga-colorization-v2 ──► Color Image
    │
    └──► Apple Depth Pro ──► Depth Map
                                  │
                                  ▼
                        WebGL Parallax Reader
                        (color + depth = 3D!)
```

### Web Transform Flow

```
Browser UI ──upload──► Express Server ──spawn──► Python Pipeline
                                                      │
     Auto-load in 3D viewer ◄──poll status──── JSON progress
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload B&W images (multipart) |
| `POST` | `/api/transform` | Start pipeline `{jobId, steps, device}` |
| `GET` | `/api/transform/:id` | Poll job status + progress |
| `GET` | `/api/outputs` | List available output folders |
| `POST` | `/api/load-folder` | Load processed pages for viewer |

## Performance

On Apple Silicon (MPS):

| Step | Time/Page | Memory |
|------|-----------|--------|
| Colorization | ~3s | ~1GB |
| Depth estimation | ~14s | ~2GB |
| **Total** | **~17s** | **~3GB** |

## Credits

- [manga-colorization-v2](https://github.com/qweasdd/manga-colorization-v2) — AI manga colorization
- [Apple Depth Pro](https://github.com/apple/ml-depth-pro) — Monocular depth estimation

## License

MIT
