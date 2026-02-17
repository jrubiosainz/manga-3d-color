# Docs

## Pipeline

```
┌─────────────┐     ┌──────────────────────┐     ┌──────────────────┐
│  B&W Manga  │────▶│ manga-colorization-v2│────▶│  *_color.png     │
│  (input)    │     │  (~3s/page on MPS)   │     │  (colorized)     │
└─────────────┘     └──────────────────────┘     └────────┬─────────┘
                                                          │
                    ┌──────────────────────┐              │
                    │  Apple Depth Pro     │◀─────────────┘
                    │  (~14s/page on MPS)  │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  *_depth.png         │
                    │  (depth map)         │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  WebGL 3D Reader     │
                    │  color + depth =     │
                    │  real-time parallax! │
                    └──────────────────────┘
```

## Reader Modes

| Mode | Description |
|------|-------------|
| **Parallax** | Mouse-tracked depth displacement (default) |
| **Layers** | Quantized depth layers with parallax |
| **Depth** | Viridis-colored depth map visualization |
| **Color** | Original colorized image |
| **Side-by-side** | Stereo pair for cross-eye 3D viewing |

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `←` `→` | Previous / Next page |
| `Space` | Toggle auto-move |
| `F` | Toggle fullscreen |
