#!/usr/bin/env python3
"""
Parallax Video Exporter
=======================
Generate smooth parallax animation videos from color + depth image pairs.
Perfect for sharing manga 3D results on social media.

Usage:
    python export_video.py color.png depth.png -o output.mp4
    python export_video.py color.png depth.png --gif -o output.gif
    python export_video.py output_dir/ --all  # Process all pairs in dir
"""

import os
import sys
import argparse
import subprocess
import tempfile
import shutil
import numpy as np
from PIL import Image
import math


def create_parallax_frame(color_arr, depth_arr, shift_x, shift_y, max_shift=30):
    """Render a single parallax frame by shifting pixels based on depth."""
    h, w, c = color_arr.shape
    
    # Normalize depth to 0-1
    depth_norm = depth_arr.astype(np.float32) / 255.0
    
    # Create displacement maps
    dx = (depth_norm * shift_x).astype(np.int32)
    dy = (depth_norm * shift_y).astype(np.int32)
    
    # Create coordinate grids
    yy, xx = np.mgrid[0:h, 0:w]
    
    # Apply displacement
    src_x = np.clip(xx - dx, 0, w - 1)
    src_y = np.clip(yy - dy, 0, h - 1)
    
    # Remap pixels
    frame = color_arr[src_y, src_x]
    
    return frame


def generate_parallax_video(color_path, depth_path, output_path,
                            duration=4.0, fps=30, max_shift=25,
                            motion='circle', resolution=None, as_gif=False):
    """
    Generate a parallax animation video/gif.
    
    Args:
        color_path: Path to colorized image
        depth_path: Path to depth map image
        output_path: Output video/gif path
        duration: Animation duration in seconds
        fps: Frames per second
        max_shift: Maximum pixel displacement
        motion: Camera motion pattern ('circle', 'figure8', 'horizontal', 'breathe')
        resolution: Output resolution as (w, h) or None for original
        as_gif: Output as GIF instead of MP4
    """
    # Load images
    color_img = Image.open(color_path).convert('RGB')
    depth_img = Image.open(depth_path).convert('L')
    
    # Resize depth to match color
    if depth_img.size != color_img.size:
        depth_img = depth_img.resize(color_img.size, Image.BILINEAR)
    
    # Optional resize
    if resolution:
        color_img = color_img.resize(resolution, Image.LANCZOS)
        depth_img = depth_img.resize(resolution, Image.BILINEAR)
    
    color_arr = np.array(color_img)
    depth_arr = np.array(depth_img)
    
    total_frames = int(duration * fps)
    h, w = color_arr.shape[:2]
    
    # Ensure even dimensions for h264
    if not as_gif:
        w_out = w if w % 2 == 0 else w - 1
        h_out = h if h % 2 == 0 else h - 1
    else:
        w_out, h_out = w, h
    
    # Create temp directory for frames
    tmp_dir = tempfile.mkdtemp(prefix='parallax_')
    
    try:
        print(f"  Rendering {total_frames} frames ({motion} motion, {duration}s @ {fps}fps)...")
        
        for i in range(total_frames):
            t = i / total_frames  # 0 to 1
            angle = t * 2 * math.pi
            
            if motion == 'circle':
                shift_x = math.sin(angle) * max_shift
                shift_y = math.cos(angle) * max_shift * 0.6
            elif motion == 'figure8':
                shift_x = math.sin(angle) * max_shift
                shift_y = math.sin(angle * 2) * max_shift * 0.4
            elif motion == 'horizontal':
                shift_x = math.sin(angle) * max_shift
                shift_y = 0
            elif motion == 'breathe':
                # Gentle zoom-like pulse
                scale = math.sin(angle) * 0.5 + 0.5  # 0 to 1
                shift_x = math.sin(angle * 2) * max_shift * 0.3
                shift_y = scale * max_shift * 0.5
            else:
                shift_x = math.sin(angle) * max_shift
                shift_y = math.cos(angle) * max_shift * 0.6
            
            frame = create_parallax_frame(color_arr, depth_arr, shift_x, shift_y, max_shift)
            
            # Crop to even dimensions
            frame = frame[:h_out, :w_out]
            
            frame_img = Image.fromarray(frame)
            frame_img.save(os.path.join(tmp_dir, f'frame_{i:05d}.png'))
            
            if (i + 1) % (total_frames // 4) == 0:
                print(f"    {i+1}/{total_frames} frames...")
        
        print(f"  Encoding {'GIF' if as_gif else 'MP4'}...")
        
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        if as_gif:
            # Generate high-quality GIF with palette
            palette_path = os.path.join(tmp_dir, 'palette.png')
            cmd_palette = [
                'ffmpeg', '-y', '-framerate', str(fps),
                '-i', os.path.join(tmp_dir, 'frame_%05d.png'),
                '-vf', f'fps={min(fps, 15)},scale={min(w_out, 480)}:-1:flags=lanczos,palettegen=stats_mode=diff',
                palette_path
            ]
            subprocess.run(cmd_palette, capture_output=True)
            
            cmd_gif = [
                'ffmpeg', '-y', '-framerate', str(fps),
                '-i', os.path.join(tmp_dir, 'frame_%05d.png'),
                '-i', palette_path,
                '-lavfi', f'fps={min(fps, 15)},scale={min(w_out, 480)}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3',
                output_path
            ]
            result = subprocess.run(cmd_gif, capture_output=True)
        else:
            cmd = [
                'ffmpeg', '-y', '-framerate', str(fps),
                '-i', os.path.join(tmp_dir, 'frame_%05d.png'),
                '-c:v', 'libx264', '-preset', 'slow', '-crf', '18',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                output_path
            ]
            result = subprocess.run(cmd, capture_output=True)
        
        if result.returncode != 0:
            print(f"  ⚠️ ffmpeg error: {result.stderr.decode()[-200:]}")
            return None
        
        size = os.path.getsize(output_path)
        print(f"  ✅ Saved: {output_path} ({size/1024:.0f}KB)")
        return output_path
        
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def process_directory(dir_path, output_dir=None, **kwargs):
    """Find all color+depth pairs in a directory and generate videos."""
    if output_dir is None:
        output_dir = dir_path
    
    pairs = []
    files = os.listdir(dir_path)
    
    for f in sorted(files):
        if '_color.' in f:
            base = f.replace('_color.', '.')
            name = os.path.splitext(f.split('_color')[0])[0]
            depth_file = f.replace('_color.', '_depth.')
            if depth_file in files:
                pairs.append({
                    'name': name,
                    'color': os.path.join(dir_path, f),
                    'depth': os.path.join(dir_path, depth_file)
                })
    
    if not pairs:
        print("No color+depth pairs found.")
        return []
    
    print(f"Found {len(pairs)} pair(s)")
    results = []
    
    for p in pairs:
        ext = '.gif' if kwargs.get('as_gif') else '.mp4'
        out = os.path.join(output_dir, f"{p['name']}_parallax{ext}")
        print(f"\n📹 {p['name']}:")
        result = generate_parallax_video(p['color'], p['depth'], out, **kwargs)
        if result:
            results.append(result)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Parallax Video Exporter')
    parser.add_argument('color', help='Color image path (or directory with --all)')
    parser.add_argument('depth', nargs='?', help='Depth map path')
    parser.add_argument('-o', '--output', help='Output path')
    parser.add_argument('--all', action='store_true', help='Process all pairs in directory')
    parser.add_argument('--gif', action='store_true', help='Output as GIF')
    parser.add_argument('--duration', type=float, default=4.0, help='Duration in seconds (default: 4)')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second (default: 30)')
    parser.add_argument('--shift', type=int, default=25, help='Max parallax shift in pixels (default: 25)')
    parser.add_argument('--motion', choices=['circle', 'figure8', 'horizontal', 'breathe'],
                        default='circle', help='Camera motion pattern (default: circle)')
    parser.add_argument('--width', type=int, help='Output width (preserves aspect ratio)')
    args = parser.parse_args()
    
    kwargs = {
        'duration': args.duration,
        'fps': args.fps,
        'max_shift': args.shift,
        'motion': args.motion,
        'as_gif': args.gif,
    }
    
    if args.all or os.path.isdir(args.color):
        dir_path = args.color
        process_directory(dir_path, args.output, **kwargs)
    else:
        if not args.depth:
            print("Error: depth map required (or use --all with a directory)")
            sys.exit(1)
        
        if not args.output:
            base = os.path.splitext(os.path.basename(args.color))[0].replace('_color', '')
            ext = '.gif' if args.gif else '.mp4'
            args.output = f"{base}_parallax{ext}"
        
        generate_parallax_video(args.color, args.depth, args.output, **kwargs)


if __name__ == '__main__':
    main()
