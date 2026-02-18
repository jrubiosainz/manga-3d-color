#!/usr/bin/env python3
"""
Manga 3D Color Pipeline
========================
Input: B&W manga page (image or PDF)
Output: Colorized image with 3D depth effect

Pipeline:
1. Extract pages from PDF (if PDF input)
2. Colorize B&W manga using manga-colorization-v2
3. Generate depth map using Apple Depth Pro
4. Create 3D parallax effect combining color + depth
5. Reassemble into PDF (if PDF input)

Usage:
    python manga_pipeline.py input.jpg              # Single image
    python manga_pipeline.py input.pdf               # Full PDF
    python manga_pipeline.py input_folder/           # Folder of images
    python manga_pipeline.py input.jpg --steps color  # Only colorize
    python manga_pipeline.py input.jpg --steps depth  # Only depth
    python manga_pipeline.py input.jpg --steps all    # Full pipeline (default)
"""

import os
import sys
import argparse
import time
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add manga-colorization-v2 to path
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLORIZER_DIR = os.path.join(SCRIPT_DIR, 'manga-colorization-v2')
sys.path.insert(0, COLORIZER_DIR)


def extract_pdf_pages(pdf_path, output_dir):
    """Extract pages from PDF as images."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("Installing PyMuPDF...")
        os.system(f"{sys.executable} -m pip install PyMuPDF -q")
        import fitz
    
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        # Render at 2x for quality
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        img_path = os.path.join(output_dir, f"page_{i+1:03d}.png")
        pix.save(img_path)
        pages.append(img_path)
        print(f"  Extracted page {i+1}/{len(doc)}")
    doc.close()
    return pages


def colorize_image(image_path, output_path, colorizer):
    """Colorize a B&W manga image."""
    image = plt.imread(image_path)
    colorizer.set_image(image, 576, True, 25)
    result = colorizer.colorize()
    plt.imsave(output_path, result)
    return output_path


def generate_depth(image_path, output_path, depth_model, depth_transform):
    """Generate depth map using Apple Depth Pro."""
    import torch
    import depth_pro
    
    # Load image using depth_pro's utility
    image_np, _, f_px = depth_pro.load_rgb(image_path)
    
    # Transform for model input
    image_tensor = depth_transform(image_np)
    
    # Run depth prediction
    with torch.no_grad():
        prediction = depth_model.infer(image_tensor.to(next(depth_model.parameters()).device), f_px=f_px)
    
    depth = prediction["depth"].squeeze().cpu().numpy()
    
    # Normalize depth to 0-255
    depth_norm = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    
    # Save depth map
    depth_img = Image.fromarray(depth_norm)
    depth_img.save(output_path)
    
    return output_path, depth_norm


def create_3d_parallax(color_path, depth_map, output_path, shift_amount=15):
    """Create a 3D parallax/anaglyph effect from color + depth."""
    color_img = Image.open(color_path).convert('RGB')
    color_arr = np.array(color_img)
    
    # Resize depth to match color
    h, w = color_arr.shape[:2]
    depth_resized = np.array(Image.fromarray(depth_map).resize((w, h)))
    
    # Normalize depth to shift range
    depth_shift = (depth_resized.astype(float) / 255.0 * shift_amount).astype(int)
    
    # Create left/right shifted views for anaglyph 3D
    left_img = np.zeros_like(color_arr)
    right_img = np.zeros_like(color_arr)
    
    for y in range(h):
        for x in range(w):
            shift = depth_shift[y, x]
            # Left eye (shift right for near objects)
            lx = min(x + shift, w - 1)
            left_img[y, lx] = color_arr[y, x]
            # Right eye (shift left for near objects)
            rx = max(x - shift, 0)
            right_img[y, rx] = color_arr[y, x]
    
    # Anaglyph: Red channel from left, Cyan from right
    anaglyph = np.zeros_like(color_arr)
    anaglyph[:, :, 0] = left_img[:, :, 0]  # Red from left
    anaglyph[:, :, 1] = right_img[:, :, 1]  # Green from right
    anaglyph[:, :, 2] = right_img[:, :, 2]  # Blue from right
    
    result = Image.fromarray(anaglyph)
    result.save(output_path)
    
    # Also save a side-by-side comparison
    comparison_path = output_path.replace('.png', '_comparison.png')
    depth_colored = Image.fromarray(
        plt.cm.viridis(depth_resized / 255.0, bytes=True)[:, :, :3]
    )
    
    # Create comparison: original | colorized | depth | 3D
    comp_w = w * 2
    comp_h = h * 2
    comparison = Image.new('RGB', (comp_w, comp_h))
    
    # Load original B&W
    original = Image.open(color_path).convert('RGB').resize((w, h))
    
    comparison.paste(original, (0, 0))
    comparison.paste(color_img, (w, 0))
    comparison.paste(depth_colored.resize((w, h)), (0, h))
    comparison.paste(result, (w, h))
    comparison.save(comparison_path, quality=90)
    
    return output_path, comparison_path


def create_pdf(image_paths, output_path):
    """Reassemble processed images into a PDF."""
    images = [Image.open(p).convert('RGB') for p in image_paths]
    if images:
        images[0].save(output_path, save_all=True, append_images=images[1:])
        print(f"  PDF saved: {output_path}")


def process_single(image_path, output_dir, colorizer, depth_model, depth_transform, steps='all'):
    """Process a single manga page through the pipeline."""
    basename = os.path.splitext(os.path.basename(image_path))[0]
    results = {}
    
    # Step 1: Colorize
    if steps in ('all', 'color'):
        color_path = os.path.join(output_dir, f"{basename}_color.png")
        print(f"  🎨 Colorizing {basename}...")
        t0 = time.time()
        colorize_image(image_path, color_path, colorizer)
        print(f"     Done in {time.time()-t0:.1f}s")
        results['color'] = color_path
    else:
        # Use input as color (assume already colored)
        results['color'] = image_path
    
    # Step 2: Depth
    if steps in ('all', 'depth'):
        depth_path = os.path.join(output_dir, f"{basename}_depth.png")
        source_for_depth = results.get('color', image_path)
        print(f"  🔮 Generating depth for {basename}...")
        t0 = time.time()
        _, depth_map = generate_depth(source_for_depth, depth_path, depth_model, depth_transform)
        print(f"     Done in {time.time()-t0:.1f}s")
        results['depth'] = depth_path
        results['depth_map'] = depth_map
    
    # Step 3: 3D Effect
    if steps == 'all' and 'depth_map' in results:
        parallax_path = os.path.join(output_dir, f"{basename}_3d.png")
        print(f"  🖼️  Creating 3D effect for {basename}...")
        t0 = time.time()
        _, comparison_path = create_3d_parallax(
            results['color'], results['depth_map'], parallax_path
        )
        print(f"     Done in {time.time()-t0:.1f}s")
        results['3d'] = parallax_path
        results['comparison'] = comparison_path
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Manga 3D Color Pipeline')
    parser.add_argument('input', help='Input image, PDF, or folder')
    parser.add_argument('-o', '--output', default=None, help='Output directory')
    parser.add_argument('-s', '--steps', choices=['all', 'color', 'depth'], default='all')
    parser.add_argument('--size', type=int, default=576, help='Colorization size')
    parser.add_argument('--no-depth', action='store_true', help='Skip depth generation')
    parser.add_argument('--viewer', action='store_true', default=True, help='Generate parallax viewer HTML (default: on)')
    parser.add_argument('--no-viewer', action='store_true', help='Skip viewer generation')
    parser.add_argument('--device', choices=['auto', 'mps', 'cuda', 'cpu'], default='auto',
                        help='Compute device (default: auto-detect MPS→CUDA→CPU)')
    parser.add_argument('--json-progress', action='store_true',
                        help='Output JSON progress lines for integration with web server')
    args = parser.parse_args()
    
    input_path = args.input
    
    # Determine output dir
    if args.output:
        output_dir = args.output
    else:
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = os.path.join(SCRIPT_DIR, 'output', base)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📚 Manga 3D Color Pipeline")
    print(f"   Input: {input_path}")
    print(f"   Output: {output_dir}")
    print(f"   Steps: {args.steps}")
    print()
    
    def progress(phase, current, total, msg=''):
        if args.json_progress:
            print(json.dumps({'phase': phase, 'current': current, 'total': total, 'msg': msg}), flush=True)
        else:
            print(msg if msg else f"  [{phase}] {current}/{total}")

    # Initialize device: MPS → CUDA → CPU
    import torch
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    else:
        device = args.device
    print(f"🔧 Device: {device}")
    progress('init', 0, 1, f'Using device: {device}')
    
    colorizer = None
    if args.steps in ('all', 'color'):
        print("🎨 Loading colorizer...")
        from colorizator import MangaColorizator
        os.chdir(COLORIZER_DIR)
        colorizer = MangaColorizator(device)
        os.chdir(SCRIPT_DIR)
        print("   Colorizer ready!")
    
    depth_model = None
    depth_transform = None
    if args.steps in ('all', 'depth') and not args.no_depth:
        print("🔮 Loading Depth Pro...")
        sys.path.insert(0, os.path.join(SCRIPT_DIR, 'ml-depth-pro'))
        import depth_pro
        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
        from dataclasses import replace
        config = replace(DEFAULT_MONODEPTH_CONFIG_DICT, 
                        checkpoint_uri=os.path.join(SCRIPT_DIR, 'checkpoints', 'depth_pro.pt'))
        depth_model, depth_transform = depth_pro.create_model_and_transforms(
            config=config, device=device
        )
        depth_model.eval()
        print("   Depth Pro ready!")
    
    print()
    
    # Collect input images
    image_paths = []
    is_pdf = False
    
    if os.path.isdir(input_path):
        for f in sorted(os.listdir(input_path)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(input_path, f))
    elif input_path.lower().endswith('.pdf'):
        is_pdf = True
        pdf_pages_dir = os.path.join(output_dir, 'pages')
        os.makedirs(pdf_pages_dir, exist_ok=True)
        print("📄 Extracting PDF pages...")
        image_paths = extract_pdf_pages(input_path, pdf_pages_dir)
    elif os.path.isfile(input_path):
        image_paths = [input_path]
    else:
        print(f"Error: {input_path} not found")
        sys.exit(1)
    
    print(f"📑 Processing {len(image_paths)} page(s)...\n")
    
    all_results = []
    for i, img_path in enumerate(image_paths):
        print(f"[{i+1}/{len(image_paths)}] {os.path.basename(img_path)}")
        results = process_single(
            img_path, output_dir, colorizer, depth_model, depth_transform, args.steps
        )
        all_results.append(results)
        print()
    
    # Reassemble PDF if input was PDF
    if is_pdf and args.steps == 'all':
        color_pages = [r['color'] for r in all_results if 'color' in r]
        if color_pages:
            pdf_color_path = os.path.join(output_dir, 
                os.path.splitext(os.path.basename(input_path))[0] + '_color.pdf')
            print("📕 Creating color PDF...")
            create_pdf(color_pages, pdf_color_path)
    
    # Generate parallax viewer for single-image results
    if not args.no_viewer and args.steps == 'all' and len(all_results) == 1:
        r = all_results[0]
        if 'color' in r and 'depth' in r:
            try:
                from generate_viewer import generate_viewer as gen_viewer
                basename = os.path.splitext(os.path.basename(image_paths[0]))[0]
                viewer_path = os.path.join(output_dir, f"{basename}_parallax.html")
                print("🌌 Generating parallax viewer...")
                gen_viewer(r['color'], r['depth'], viewer_path, basename.replace('_', ' ').title())
            except Exception as e:
                print(f"   ⚠️ Viewer generation failed: {e}")
    
    # Batch viewer: generate for each result + multi-page reader
    if not args.no_viewer and args.steps == 'all' and len(all_results) > 1:
        print("🌌 Generating parallax viewers...")
        try:
            from generate_viewer import generate_viewer as gen_viewer
            for i, r in enumerate(all_results):
                if 'color' in r and 'depth' in r:
                    basename = os.path.splitext(os.path.basename(image_paths[i]))[0]
                    viewer_path = os.path.join(output_dir, f"{basename}_parallax.html")
                    gen_viewer(r['color'], r['depth'], viewer_path, basename.replace('_', ' ').title())
        except Exception as e:
            print(f"   ⚠️ Batch viewer generation failed: {e}")
        
        # Generate multi-page 3D reader
        try:
            from generate_reader import generate_reader, find_pairs
            input_name = os.path.splitext(os.path.basename(input_path))[0]
            reader_path = os.path.join(output_dir, f"{input_name}_reader.html")
            # Build pairs from results
            pairs = []
            for i, r in enumerate(all_results):
                if 'color' in r and 'depth' in r:
                    basename = os.path.splitext(os.path.basename(image_paths[i]))[0]
                    pairs.append({'name': basename, 'color': r['color'], 'depth': r['depth']})
            if pairs:
                print("📖 Generating multi-page 3D reader...")
                generate_reader(pairs, reader_path, input_name.replace('_', ' ').title())
        except Exception as e:
            print(f"   ⚠️ Reader generation failed: {e}")
    
    print("✅ Pipeline complete!")
    print(f"   Output: {output_dir}")
    
    # List output files
    for f in sorted(os.listdir(output_dir)):
        if not os.path.isdir(os.path.join(output_dir, f)):
            size = os.path.getsize(os.path.join(output_dir, f))
            print(f"   📄 {f} ({size/1024:.0f}KB)")


if __name__ == '__main__':
    main()
