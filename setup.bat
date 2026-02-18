@echo off
echo ========================================
echo  Manga 3D Color - Windows Setup
echo ========================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Install Python 3.9+ from python.org
    pause
    exit /b 1
)

:: Check Node.js
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Node.js not found. Install from nodejs.org
    pause
    exit /b 1
)

echo [1/4] Creating conda environment...
conda create -n manga3d python=3.11 -y
call conda activate manga3d

echo [2/4] Installing PyTorch (CUDA)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo [3/4] Installing Python dependencies...
pip install -r requirements.txt

echo [4/4] Installing Node.js dependencies...
cd reader
npm install
cd ..

echo.
echo ========================================
echo  Setup complete!
echo.
echo  To process manga:
echo    conda activate manga3d
echo    python pipeline\manga_pipeline.py input.jpg
echo.
echo  To start the 3D reader:
echo    cd reader ^&^& npm start
echo    Open http://localhost:3002
echo ========================================
pause
