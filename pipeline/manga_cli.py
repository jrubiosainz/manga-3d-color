#!/usr/bin/env python3
"""
Manga 3D Color — CLI Wrapper v2
=================================
One-command pipeline: B&W manga → Colorized + 3D Depth + Reader.

v2 improvements:
- Batched pipeline: colorize ALL → depth ALL (less model switching overhead)
- Parallel CPU work: thumbnails, PDF assembly via thread pool
- Progress callbacks for web UI integration
- Live throughput & ETA tracking
- --workers flag for CPU parallelism
- Memory management: explicit cleanup between GPU phases

Supports: single image, folder of images, PDF (full round-trip).

Usage:
    python manga_cli.py manga.pdf                      # Full pipeline, PDF output
    python manga_cli.py manga.pdf --preset draft        # Fast preview (lower res)
    python manga_cli.py manga.pdf --preset high         # Max quality
    python manga_cli.py page.jpg                        # Single image
    python manga_cli.py pages/                           # Folder of images
    python manga_cli.py manga.pdf --steps color          # Only colorize
    python manga_cli.py manga.pdf --no-reader            # Skip HTML reader
    python manga_cli.py manga.pdf --compare              # Side-by-side PDF (B&W | Color)
    python manga_cli.py manga.pdf --workers 4            # Parallel CPU workers
    python manga_cli.py page.jpg --video                 # Generate parallax MP4
    python manga_cli.py manga.pdf --video-gif            # Generate parallax GIFs

Outputs:
    output/<name>/
    ├── <name>_color.pdf        # Colorized PDF (if input was PDF)
    ├── <name>_compare.pdf      # Side-by-side B&W vs Color (--compare)
    ├── <name>_reader.html      # Interactive 3D reader
    ├── page_001_color.png      # Per-page color
    ├── page_001_depth.png      # Per-page depth map
    └── thumbnails/             # Optimized thumbnails for reader

Requirements:
    conda activate manga3d
"""

import os
import sys
import gc
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COLORIZER_DIR = os.path.join(SCRIPT_DIR, 'manga-colorization-v2')
sys.path.insert(0, COLORIZER_DIR)

# ── Presets ──────────────────────────────────────────────────────────

PRESETS = {
    'draft': {
        'color_size': 384,
        'depth_enabled': True,
        'pdf_dpi': 150,
        'thumb_width': 120,
        'jpg_quality': 75,
        'desc': 'Fast preview — lower resolution, ~2x faster'
    },
    'normal': {
        'color_size': 576,
        'depth_enabled': True,
        'pdf_dpi': 200,
        'thumb_width': 160,
        'jpg_quality': 88,
        'desc': 'Balanced quality and speed (default)'
    },
    'high': {
        'color_size': 768,
        'depth_enabled': True,
        'pdf_dpi': 300,
        'thumb_width': 200,
        'jpg_quality': 95,
        'desc': 'Maximum quality — slower processing'
    }
}


# ── Helpers ──────────────────────────────────────────────────────────

def fmt_time(seconds):
    """Format seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


def fmt_size(bytes_val):
    """Format bytes to human-readable."""
    if bytes_val < 1024:
        return f"{bytes_val}B"
    elif bytes_val < 1024 * 1024:
        return f"{bytes_val/1024:.0f}KB"
    else:
        return f"{bytes_val/1024/1024:.1f}MB"


def progress_bar(current, total, width=30):
    """Simple text progress bar."""
    pct = current / total if total > 0 else 0
    filled = int(width * pct)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}] {current}/{total}"


class Timer:
    """Context manager for timing operations."""
    def __init__(self, label=""):
        self.label = label
        self.start = None
        self.elapsed = 0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start


class ThroughputTracker:
    """Track processing speed and estimate remaining time."""

    def __init__(self, total):
        self.total = total
        self.times = []
        self.start = time.time()

    def record(self, elapsed):
        self.times.append(elapsed)

    @property
    def avg(self):
        return sum(self.times) / len(self.times) if self.times else 0

    @property
    def pages_per_min(self):
        return 60 / self.avg if self.avg > 0 else 0

    @property
    def eta(self):
        remaining = self.total - len(self.times)
        return self.avg * remaining if self.avg > 0 else 0

    def status(self, current):
        parts = [f"{self.avg:.1f}s/page"]
        if self.pages_per_min > 0:
            parts.append(f"{self.pages_per_min:.1f} pages/min")
        if current < self.total:
            parts.append(f"ETA {fmt_time(self.eta)}")
        return " · ".join(parts)


class ProgressCallback:
    """Callback interface for progress reporting (used by web UI)."""

    def __init__(self):
        self.listeners = []

    def add_listener(self, fn):
        """Add a callback: fn(phase, current, total, message)"""
        self.listeners.append(fn)

    def emit(self, phase, current, total, message=""):
        for fn in self.listeners:
            try:
                fn(phase, current, total, message)
            except Exception:
                pass

    def log(self, message):
        """Emit a log-only event."""
        self.emit("log", 0, 0, message)


# Global default (no-op unless listeners added)
_progress = ProgressCallback()


# ── PDF Operations ───────────────────────────────────────────────────

def extract_pdf(pdf_path, output_dir, dpi=200):
    """Extract pages from PDF as high-quality images."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        os.system(f"{sys.executable} -m pip install PyMuPDF -q")
        import fitz

    doc = fitz.open(pdf_path)
    pages = []
    scale = dpi / 72  # PDF base is 72 DPI
    mat = fitz.Matrix(scale, scale)

    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat)
        img_path = os.path.join(output_dir, f"page_{i+1:03d}.png")
        pix.save(img_path)
        pages.append(img_path)

    doc.close()
    return pages


def create_color_pdf(color_paths, output_path, quality=88):
    """Create a high-quality color PDF from processed images using PyMuPDF."""
    try:
        import fitz
    except ImportError:
        os.system(f"{sys.executable} -m pip install PyMuPDF -q")
        import fitz

    doc = fitz.open()

    for img_path in color_paths:
        img = fitz.open(img_path)
        pdfbytes = img.convert_to_pdf()
        img.close()
        img_pdf = fitz.open("pdf", pdfbytes)
        doc.insert_pdf(img_pdf)
        img_pdf.close()

    doc.save(output_path, deflate=True)
    doc.close()
    return output_path


def create_compare_pdf(original_paths, color_paths, output_path, quality=88):
    """Create side-by-side comparison PDF: B&W left, Color right."""
    from PIL import Image
    import tempfile

    try:
        import fitz
    except ImportError:
        os.system(f"{sys.executable} -m pip install PyMuPDF -q")
        import fitz

    doc = fitz.open()
    tmp_files = []

    for orig_path, color_path in zip(original_paths, color_paths):
        orig = Image.open(orig_path).convert('RGB')
        color = Image.open(color_path).convert('RGB')

        # Match heights
        target_h = max(orig.height, color.height)
        if orig.height != target_h:
            ratio = target_h / orig.height
            orig = orig.resize((int(orig.width * ratio), target_h), Image.LANCZOS)
        if color.height != target_h:
            ratio = target_h / color.height
            color = color.resize((int(color.width * ratio), target_h), Image.LANCZOS)

        # Gap between panels
        gap = 20
        total_w = orig.width + gap + color.width

        canvas = Image.new('RGB', (total_w, target_h), (40, 40, 40))
        canvas.paste(orig, (0, 0))
        canvas.paste(color, (orig.width + gap, 0))

        tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        canvas.save(tmp.name, 'JPEG', quality=quality, optimize=True)
        tmp_files.append(tmp.name)

        img = fitz.open(tmp.name)
        pdfbytes = img.convert_to_pdf()
        img.close()
        img_pdf = fitz.open("pdf", pdfbytes)
        doc.insert_pdf(img_pdf)
        img_pdf.close()

    doc.save(output_path, deflate=True)
    doc.close()

    for f in tmp_files:
        try:
            os.unlink(f)
        except Exception:
            pass

    return output_path


# ── Thumbnail Generation ─────────────────────────────────────────────

def _create_single_thumbnail(args):
    """Worker for parallel thumbnail generation."""
    img_path, thumb_dir, thumb_width = args
    from PIL import Image

    img = Image.open(img_path).convert('RGB')
    ratio = thumb_width / img.width
    thumb_h = int(img.height * ratio)
    img = img.resize((thumb_width, thumb_h), Image.LANCZOS)

    thumb_path = os.path.join(thumb_dir, os.path.basename(img_path).replace('.png', '_thumb.jpg'))
    img.save(thumb_path, 'JPEG', quality=70, optimize=True)
    return thumb_path


def create_thumbnails(color_paths, thumb_dir, thumb_width=160, workers=4):
    """Generate optimized thumbnails in parallel."""
    os.makedirs(thumb_dir, exist_ok=True)

    tasks = [(p, thumb_dir, thumb_width) for p in color_paths]

    if workers > 1 and len(tasks) > 1:
        thumbs = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_create_single_thumbnail, t): i for i, t in enumerate(tasks)}
            results = [None] * len(tasks)
            for fut in as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()
            thumbs = results
    else:
        thumbs = [_create_single_thumbnail(t) for t in tasks]

    return thumbs


# ── Reader ───────────────────────────────────────────────────────────

def generate_optimized_reader(pairs, output_path, title, thumb_paths=None):
    """Generate reader with optimized thumbnails (smaller file size)."""
    sys.path.insert(0, SCRIPT_DIR)
    from generate_reader import generate_reader
    generate_reader(pairs, output_path, title)
    return output_path


# ── GPU Model Management ─────────────────────────────────────────────

def load_colorizer(device, color_size=576):
    """Load colorization model only."""
    from colorizator import MangaColorizator
    cwd = os.getcwd()
    os.chdir(COLORIZER_DIR)
    colorizer = MangaColorizator(device)
    os.chdir(cwd)
    return colorizer


def unload_colorizer(colorizer):
    """Release colorizer from GPU memory."""
    del colorizer
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def load_depth_model(device):
    """Load Depth Pro model only."""
    sys.path.insert(0, os.path.join(SCRIPT_DIR, 'ml-depth-pro'))
    import depth_pro
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
    from dataclasses import replace
    config = replace(
        DEFAULT_MONODEPTH_CONFIG_DICT,
        checkpoint_uri=os.path.join(SCRIPT_DIR, 'checkpoints', 'depth_pro.pt')
    )
    model, transform = depth_pro.create_model_and_transforms(config=config, device=device)
    model.eval()
    return model, transform


def unload_depth_model(model, transform):
    """Release depth model from GPU memory."""
    del model
    del transform
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


# ── Core Processing ──────────────────────────────────────────────────

def colorize_page(image_path, output_path, colorizer, color_size=576):
    """Colorize a single B&W image."""
    import matplotlib.pyplot as plt

    image = plt.imread(image_path)
    colorizer.set_image(image, color_size, True, 25)
    result = colorizer.colorize()
    plt.imsave(output_path, result)
    return output_path


def depth_page(image_path, output_path, depth_model, depth_transform):
    """Generate depth map for a single image."""
    import torch
    import numpy as np
    from PIL import Image

    sys.path.insert(0, os.path.join(SCRIPT_DIR, 'ml-depth-pro'))
    import depth_pro

    image_np, _, f_px = depth_pro.load_rgb(image_path)
    image_tensor = depth_transform(image_np)

    with torch.no_grad():
        prediction = depth_model.infer(
            image_tensor.to(next(depth_model.parameters()).device),
            f_px=f_px
        )

    depth = prediction["depth"].squeeze().cpu().numpy()
    depth_norm = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)

    depth_img = Image.fromarray(depth_norm)
    depth_img.save(output_path)

    return output_path, depth_norm


# ── Batched Pipeline ─────────────────────────────────────────────────

def batch_colorize(image_paths, output_dir, device, color_size=576, progress=None):
    """Colorize all pages in one batch (single model load)."""
    if progress is None:
        progress = _progress

    total = len(image_paths)
    tracker = ThroughputTracker(total)
    results = {}

    print("  🎨 Phase 1: Colorization")
    progress.emit("color", 0, total, "Loading colorizer...")

    with Timer("Colorizer load") as t:
        colorizer = load_colorizer(device, color_size)
    print(f"     Model loaded in {fmt_time(t.elapsed)}")

    for i, img_path in enumerate(image_paths):
        basename = f"page_{i+1:03d}"
        color_path = os.path.join(output_dir, f"{basename}_color.png")
        bar = progress_bar(i + 1, total, 25)

        t0 = time.time()
        colorize_page(img_path, color_path, colorizer, color_size)
        elapsed = time.time() - t0
        tracker.record(elapsed)

        results[img_path] = color_path
        status = tracker.status(i + 1)
        print(f"     {bar} {basename} — {elapsed:.1f}s ({status})")
        progress.emit("color", i + 1, total, f"{basename} — {elapsed:.1f}s")

    # Free GPU memory before depth phase
    print("     Unloading colorizer...")
    unload_colorizer(colorizer)

    print(f"  ✓ Colorization done — avg {tracker.avg:.1f}s/page, {tracker.pages_per_min:.1f} pages/min\n")
    return results


def batch_depth(color_paths_map, output_dir, device, progress=None):
    """Generate depth maps for all pages in one batch (single model load)."""
    if progress is None:
        progress = _progress

    # color_paths_map: {original_path: color_path}
    total = len(color_paths_map)
    tracker = ThroughputTracker(total)
    results = {}

    print("  🔮 Phase 2: Depth Estimation")
    progress.emit("depth", 0, total, "Loading Depth Pro...")

    with Timer("Depth load") as t:
        depth_model, depth_transform = load_depth_model(device)
    print(f"     Model loaded in {fmt_time(t.elapsed)}")

    for i, (orig_path, color_path) in enumerate(color_paths_map.items()):
        basename = f"page_{i+1:03d}"
        depth_path = os.path.join(output_dir, f"{basename}_depth.png")
        bar = progress_bar(i + 1, total, 25)

        t0 = time.time()
        _, depth_map = depth_page(color_path, depth_path, depth_model, depth_transform)
        elapsed = time.time() - t0
        tracker.record(elapsed)

        results[orig_path] = {'depth': depth_path, 'depth_map': depth_map}
        status = tracker.status(i + 1)
        print(f"     {bar} {basename} — {elapsed:.1f}s ({status})")
        progress.emit("depth", i + 1, total, f"{basename} — {elapsed:.1f}s")

    # Free GPU memory
    print("     Unloading depth model...")
    unload_depth_model(depth_model, depth_transform)

    print(f"  ✓ Depth done — avg {tracker.avg:.1f}s/page, {tracker.pages_per_min:.1f} pages/min\n")
    return results


# ── Output Generation (parallel-capable) ─────────────────────────────

def generate_outputs(all_results, input_path, output_dir, preset, args, workers=4, progress=None):
    """Generate all output files (PDFs, thumbnails, reader) with parallel CPU work."""
    if progress is None:
        progress = _progress

    output_files = []
    base_name = Path(input_path).stem
    is_pdf = input_path.lower().endswith('.pdf')

    color_pages = [r['color'] for r in all_results if 'color' in r]
    originals = [r['original'] for r in all_results]
    total_pages = len(all_results)

    # Use thread pool for CPU-bound output tasks
    futures = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Color PDF (CPU-bound, can run in parallel with thumbnails)
        if is_pdf and not args.no_pdf and args.steps in ('all', 'color') and color_pages:
            pdf_path = os.path.join(output_dir, f"{base_name}_color.pdf")
            progress.emit("output", 0, 3, "Creating color PDF...")
            print("  📕 Creating color PDF...")
            futures['color_pdf'] = pool.submit(
                create_color_pdf, color_pages, pdf_path, preset['jpg_quality']
            )

        # Comparison PDF
        if args.compare and is_pdf and originals and color_pages and len(originals) == len(color_pages):
            compare_path = os.path.join(output_dir, f"{base_name}_compare.pdf")
            progress.emit("output", 0, 3, "Creating comparison PDF...")
            print("  📊 Creating comparison PDF...")
            futures['compare_pdf'] = pool.submit(
                create_compare_pdf, originals, color_pages, compare_path, preset['jpg_quality']
            )

        # Thumbnails (parallel internally too)
        if color_pages and not args.no_reader:
            thumb_dir = os.path.join(output_dir, 'thumbnails')
            progress.emit("output", 0, 3, "Generating thumbnails...")
            print("  🖼️  Generating thumbnails...")
            futures['thumbnails'] = pool.submit(
                create_thumbnails, color_pages, thumb_dir, preset['thumb_width'], workers
            )

        # Collect results
        for key, fut in futures.items():
            try:
                result = fut.result()
                if key == 'color_pdf':
                    size = os.path.getsize(result)
                    print(f"     ✓ Color PDF ({fmt_size(size)})")
                    output_files.append(('Color PDF', result, size))
                elif key == 'compare_pdf':
                    size = os.path.getsize(result)
                    print(f"     ✓ Compare PDF ({fmt_size(size)})")
                    output_files.append(('Compare PDF', result, size))
                elif key == 'thumbnails':
                    total_thumb_size = sum(os.path.getsize(t) for t in result)
                    print(f"     ✓ {len(result)} thumbnails ({fmt_size(total_thumb_size)})")
            except Exception as e:
                print(f"     ❌ {key} failed: {e}")

    # 3D Reader (needs thumbnails done first, so sequential)
    if not args.no_reader and args.steps == 'all' and total_pages > 0:
        pairs = []
        for r in all_results:
            if 'color' in r and 'depth' in r:
                basename = os.path.splitext(os.path.basename(r['color']))[0].replace('_color', '')
                pairs.append({'name': basename, 'color': r['color'], 'depth': r['depth']})

        if pairs:
            title = args.title or base_name.replace('_', ' ').title()
            reader_path = os.path.join(output_dir, f"{base_name}_reader.html")
            print("  📖 Generating 3D reader...")
            progress.emit("output", 2, 3, "Generating 3D reader...")
            with Timer("Reader") as t:
                generate_optimized_reader(pairs, reader_path, title)
            size = os.path.getsize(reader_path)
            print(f"     ✓ 3D Reader ({fmt_size(size)}) in {fmt_time(t.elapsed)}")
            output_files.append(('3D Reader', reader_path, size))

    progress.emit("output", 3, 3, "Done")
    return output_files


# ── Programmatic API ─────────────────────────────────────────────────

def run_pipeline(input_path, output_dir=None, preset_name='normal', steps='all',
                 compare=False, no_reader=False, no_pdf=False, title=None,
                 page_range=None, workers=4, progress_callback=None):
    """
    Run the full pipeline programmatically.

    Args:
        input_path: Path to image, folder, or PDF
        output_dir: Output directory (auto-generated if None)
        preset_name: 'draft', 'normal', or 'high'
        steps: 'all', 'color', or 'depth'
        compare: Generate comparison PDF
        no_reader: Skip HTML reader
        no_pdf: Skip PDF output
        title: Reader title
        page_range: Set of page numbers to process (1-indexed)
        workers: Number of CPU workers for parallel tasks
        progress_callback: fn(phase, current, total, message) for progress

    Returns:
        dict with keys: output_dir, pages, files, timings, errors
    """
    import torch

    preset = PRESETS[preset_name]
    progress = ProgressCallback()
    if progress_callback:
        progress.add_listener(progress_callback)

    # Resolve output dir
    if output_dir is None:
        base = Path(input_path).stem
        output_dir = os.path.join(SCRIPT_DIR, 'output', base)
    os.makedirs(output_dir, exist_ok=True)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    total_timer = time.time()
    timings = {}
    errors = []

    # ── Collect input images ─────────────────────────────────────
    image_paths = []
    is_pdf = False

    if os.path.isdir(input_path):
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        for f in sorted(os.listdir(input_path)):
            if f.lower().endswith(exts):
                image_paths.append(os.path.join(input_path, f))
    elif input_path.lower().endswith('.pdf'):
        is_pdf = True
        pages_dir = os.path.join(output_dir, '_pages')
        os.makedirs(pages_dir, exist_ok=True)
        progress.emit("extract", 0, 1, "Extracting PDF pages...")
        with Timer() as t:
            image_paths = extract_pdf(input_path, pages_dir, dpi=preset['pdf_dpi'])
        timings['extract'] = t.elapsed
    elif os.path.isfile(input_path):
        image_paths = [input_path]
    else:
        return {'error': f'Input not found: {input_path}'}

    # Apply page range filter
    if page_range:
        image_paths = [p for i, p in enumerate(image_paths) if (i + 1) in page_range]

    if not image_paths:
        return {'error': 'No images to process'}

    total_pages = len(image_paths)

    # ── Batched GPU phases ───────────────────────────────────────

    # Phase 1: Colorization (all pages, single model load)
    color_map = {}  # original_path → color_path
    if steps in ('all', 'color'):
        with Timer() as t:
            color_map = batch_colorize(image_paths, output_dir, device,
                                       preset['color_size'], progress)
        timings['color'] = t.elapsed

    # Phase 2: Depth estimation (all pages, single model load)
    depth_map = {}  # original_path → {depth, depth_map}
    if steps in ('all', 'depth') and color_map:
        with Timer() as t:
            depth_map = batch_depth(color_map, output_dir, device, progress)
        timings['depth'] = t.elapsed

    # ── Merge results ────────────────────────────────────────────

    all_results = []
    for i, img_path in enumerate(image_paths):
        r = {'original': img_path}
        if img_path in color_map:
            r['color'] = color_map[img_path]
        if img_path in depth_map:
            r['depth'] = depth_map[img_path]['depth']
            r['depth_map'] = depth_map[img_path]['depth_map']
        all_results.append(r)

    # ── Parallel output generation ───────────────────────────────

    # Create a namespace for args-like access
    class Args:
        pass
    fake_args = Args()
    fake_args.no_pdf = no_pdf
    fake_args.no_reader = no_reader
    fake_args.compare = compare
    fake_args.steps = steps
    fake_args.title = title

    with Timer() as t:
        output_files = generate_outputs(
            all_results, input_path, output_dir, preset, fake_args, workers, progress
        )
    timings['output'] = t.elapsed

    total_elapsed = time.time() - total_timer
    timings['total'] = total_elapsed

    return {
        'output_dir': output_dir,
        'pages': total_pages,
        'files': output_files,
        'timings': timings,
        'errors': errors,
    }


# ── Main CLI ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='🎨 Manga 3D Color — Transform B&W manga into colorized 3D',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s manga.pdf                  Full pipeline with PDF output
  %(prog)s manga.pdf --preset draft   Fast preview mode
  %(prog)s manga.pdf --compare        Add side-by-side comparison PDF
  %(prog)s page.jpg                   Process single image
  %(prog)s pages/ --no-reader         Skip HTML reader generation
  %(prog)s manga.pdf --steps color    Only colorize (no depth/3D)
  %(prog)s manga.pdf --workers 4      Use 4 CPU workers for parallel tasks
        """
    )
    parser.add_argument('input', help='Input: image, folder, or PDF')
    parser.add_argument('-o', '--output', help='Output directory (default: output/<name>/)')
    parser.add_argument('-p', '--preset', choices=['draft', 'normal', 'high'], default='normal',
                        help='Quality preset (default: normal)')
    parser.add_argument('-s', '--steps', choices=['all', 'color', 'depth'], default='all',
                        help='Pipeline steps to run')
    parser.add_argument('--compare', action='store_true',
                        help='Generate side-by-side comparison PDF')
    parser.add_argument('--no-reader', action='store_true',
                        help='Skip HTML 3D reader generation')
    parser.add_argument('--no-pdf', action='store_true',
                        help='Skip PDF output generation')
    parser.add_argument('--title', help='Title for the reader')
    parser.add_argument('--pages', help='Page range to process (e.g., 1-5, 1,3,5)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of CPU workers for parallel tasks (default: 4)')
    parser.add_argument('--video', action='store_true',
                        help='Generate parallax animation video (MP4)')
    parser.add_argument('--video-gif', action='store_true',
                        help='Generate parallax animation GIF')
    parser.add_argument('--video-motion', choices=['circle', 'figure8', 'horizontal', 'breathe'],
                        default='circle', help='Video camera motion pattern (default: circle)')
    parser.add_argument('--video-duration', type=float, default=4.0,
                        help='Video duration in seconds (default: 4)')
    args = parser.parse_args()

    input_path = args.input
    preset = PRESETS[args.preset]

    # Resolve output dir
    if args.output:
        output_dir = args.output
    else:
        base = Path(input_path).stem
        output_dir = os.path.join(SCRIPT_DIR, 'output', base)

    os.makedirs(output_dir, exist_ok=True)

    # Parse page range
    page_range = None
    if args.pages:
        page_range = set()
        for part in args.pages.split(','):
            if '-' in part:
                a, b = part.split('-')
                page_range.update(range(int(a), int(b) + 1))
            else:
                page_range.add(int(part))

    # ── Banner ───────────────────────────────────────────────────

    print()
    print("╔═══════════════════════════════════════════════╗")
    print("║  🎨  Manga 3D Color Pipeline v2              ║")
    print("╚═══════════════════════════════════════════════╝")
    print(f"  Input:   {input_path}")
    print(f"  Output:  {output_dir}")
    print(f"  Preset:  {args.preset} — {preset['desc']}")
    print(f"  Steps:   {args.steps}")
    print(f"  Workers: {args.workers}")
    if page_range:
        print(f"  Pages:   {sorted(page_range)}")
    print()

    import torch
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"  ⚡ Device: {device.upper()}")

    total_timer = time.time()

    # ── Collect input images ─────────────────────────────────────

    image_paths = []
    is_pdf = False

    if os.path.isdir(input_path):
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        for f in sorted(os.listdir(input_path)):
            if f.lower().endswith(exts):
                image_paths.append(os.path.join(input_path, f))
    elif input_path.lower().endswith('.pdf'):
        is_pdf = True
        pages_dir = os.path.join(output_dir, '_pages')
        os.makedirs(pages_dir, exist_ok=True)
        print(f"  📄 Extracting PDF pages (DPI: {preset['pdf_dpi']})...")
        with Timer() as t:
            image_paths = extract_pdf(input_path, pages_dir, dpi=preset['pdf_dpi'])
        print(f"     {len(image_paths)} pages extracted in {fmt_time(t.elapsed)}")
    elif os.path.isfile(input_path):
        image_paths = [input_path]
    else:
        print(f"  ❌ Input not found: {input_path}")
        sys.exit(1)

    # Apply page range filter
    if page_range:
        image_paths = [p for i, p in enumerate(image_paths) if (i + 1) in page_range]

    total_pages = len(image_paths)
    if total_pages == 0:
        print("  ❌ No images to process")
        sys.exit(1)

    print(f"\n  📑 Processing {total_pages} page(s) — batched pipeline\n")

    # ── Phase 1: Batch Colorization ──────────────────────────────

    color_map = {}
    if args.steps in ('all', 'color'):
        color_map = batch_colorize(image_paths, output_dir, device, preset['color_size'])

    # ── Phase 2: Batch Depth ─────────────────────────────────────

    depth_results = {}
    if args.steps in ('all', 'depth') and color_map:
        depth_results = batch_depth(color_map, output_dir, device)

    # ── Merge results ────────────────────────────────────────────

    all_results = []
    for i, img_path in enumerate(image_paths):
        r = {'original': img_path}
        if img_path in color_map:
            r['color'] = color_map[img_path]
        if img_path in depth_results:
            r['depth'] = depth_results[img_path]['depth']
            r['depth_map'] = depth_results[img_path]['depth_map']
        all_results.append(r)

    # ── Phase 3: Parallel Output Generation ──────────────────────

    print("  📦 Phase 3: Output Generation (parallel)\n")
    output_files = generate_outputs(all_results, input_path, output_dir, preset, args, args.workers)

    # ── Phase 4: Video Export (optional) ────────────────────────

    if args.video or args.video_gif:
        print("  🎬 Phase 4: Parallax Video Export\n")
        try:
            from export_video import generate_parallax_video
            for r in all_results:
                if 'color' in r and 'depth' in r:
                    base = Path(r['color']).stem.replace('_color', '')
                    as_gif = args.video_gif
                    ext = '.gif' if as_gif else '.mp4'
                    out_path = os.path.join(output_dir, f"{base}_parallax{ext}")
                    generate_parallax_video(
                        r['color'], r['depth'], out_path,
                        duration=args.video_duration, fps=24,
                        motion=args.video_motion, as_gif=as_gif
                    )
        except Exception as e:
            print(f"  ⚠️ Video export failed: {e}")

    # ── Summary ──────────────────────────────────────────────────

    total_elapsed = time.time() - total_timer

    print()
    print("╔═══════════════════════════════════════════════╗")
    print("║  ✅  Pipeline Complete                        ║")
    print("╚═══════════════════════════════════════════════╝")
    print(f"  Pages processed:  {total_pages}")
    print(f"  Total time:       {fmt_time(total_elapsed)}")
    print(f"  CPU workers:      {args.workers}")
    print(f"\n  📁 Output: {output_dir}")

    # List all output files
    print()
    for f in sorted(os.listdir(output_dir)):
        fp = os.path.join(output_dir, f)
        if os.path.isfile(fp):
            print(f"     📄 {f} ({fmt_size(os.path.getsize(fp))})")
        elif os.path.isdir(fp) and f != '_pages':
            count = len(os.listdir(fp))
            print(f"     📁 {f}/ ({count} files)")

    print()


if __name__ == '__main__':
    main()
