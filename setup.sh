#!/bin/bash
# Manga 3D Color — Quick Setup
set -e

echo "🎨 Manga 3D Color — Setup"
echo "=========================="

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "❌ Python 3 required. Install it first."
    exit 1
fi

# Check Node
if ! command -v node &>/dev/null; then
    echo "⚠️  Node.js not found. Reader won't work without it."
fi

# Python deps
echo ""
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Clone colorizer
if [ ! -d "manga-colorization-v2" ]; then
    echo ""
    echo "🎨 Cloning manga-colorization-v2..."
    git clone https://github.com/qweasdd/manga-colorization-v2.git
    echo "⚠️  Download the colorizer model weights — see manga-colorization-v2/README.md"
fi

# Clone Depth Pro
if [ ! -d "ml-depth-pro" ]; then
    echo ""
    echo "🔮 Cloning Apple Depth Pro..."
    git clone https://github.com/apple/ml-depth-pro.git
    cd ml-depth-pro && pip install -e . && cd ..
    echo "⚠️  Download depth_pro.pt weights — see ml-depth-pro/README.md"
    echo "   Place at: checkpoints/depth_pro.pt"
    mkdir -p checkpoints
fi

# Reader deps
if command -v node &>/dev/null; then
    echo ""
    echo "📖 Installing reader dependencies..."
    cd reader && npm install && cd ..
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Usage:"
echo "  python pipeline/manga_pipeline.py your_manga.jpg"
echo "  cd reader && npm start"
echo "  Open http://localhost:3002"
