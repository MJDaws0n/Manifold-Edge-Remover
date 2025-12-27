#!/bin/bash

# Build script for creating a standalone Linux executable

set -e  # Exit on error

echo "================================================"
echo "Building STL Fixer for Linux"
echo "================================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Running setup..."
    ./setup_linux.sh
fi

# Activate virtual environment
source venv/bin/activate

# Install PyInstaller if not already installed
echo "Installing PyInstaller..."
pip install pyinstaller

echo ""

# Clean previous builds
if [ -d "build" ]; then
    echo "Cleaning previous build..."
    rm -rf build
fi

if [ -d "dist" ]; then
    echo "Cleaning previous dist..."
    rm -rf dist
fi

echo ""

# Build the executable
echo "Building executable..."
pyinstaller --onefile \
            --windowed \
            --name STL_Fixer \
            stl_fixer.py

echo ""
echo "================================================"
echo "✓ Build complete!"
echo "================================================"
echo ""
echo "The standalone application is located at:"
echo "  dist/STL_Fixer"
echo ""
echo "You can move this file anywhere and run it without Python installed."
echo ""
echo "Note: The executable may require certain system libraries to be installed:"
echo "  - libgtk-3-0 (for GUI)"
echo "  - libtk8.6 (for tkinter)"
echo ""
