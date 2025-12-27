# Quick Start Guide

Get started with STL Manifold Edge Fixer in minutes!

## First Time Setup

### macOS

1. Open Terminal
2. Navigate to the application folder:
   ```bash
   cd path/to/Manifold-Edge-Remover
   ```
3. Run the setup script:
   ```bash
   ./setup_mac.sh
   ```

### Windows

1. Open the folder in File Explorer
2. Double-click `setup_windows.bat`
3. Wait for installation to complete

### Linux

1. Open Terminal
2. Navigate to the application folder:
   ```bash
   cd path/to/Manifold-Edge-Remover
   ```
3. Run the setup script:
   ```bash
   ./setup_linux.sh
   ```

## Running the Application

### macOS

Double-click `run_mac.sh` or run in Terminal:
```bash
./run_mac.sh
```

### Windows

Double-click `run_windows.bat`

### Linux

Double-click `run_linux.sh` or run in Terminal:
```bash
./run_linux.sh
```

## Using the Application

1. When the file dialog appears, select your STL file
2. Wait for processing to complete (you'll see progress in the console)
3. Find your fixed file in the same directory with `_FIXED` added to the name
   - Example: `my_model.stl` → `my_model_FIXED.stl`

## Creating a Standalone Executable

If you want an application that doesn't require Python:

### macOS
```bash
./build_mac.sh
```
Find the app in `dist/STL_Fixer`

### Windows
Double-click `build_windows.bat` or run:
```cmd
build_windows.bat
```
Find the app in `dist\STL_Fixer.exe`

### Linux
```bash
./build_linux.sh
```
Find the app in `dist/STL_Fixer`

## Troubleshooting

**"Permission denied" on macOS/Linux:**
```bash
chmod +x *.sh
```

**"No module named install" after running `python -m install ...`:**
```bash
python -m pip install scipy networkx
```

**Slicer still reports non-manifold edges after fixing:**
- Run setup again to ensure all dependencies are installed (the app uses extra libraries for robust watertight repair):
```bash
python -m pip install -r requirements.txt
```

**"Python not found" on Windows:**
- Download Python from [python.org](https://www.python.org/downloads/)
- During installation, check "Add Python to PATH"

**"tkinter not found" on Linux:**
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# Fedora
sudo dnf install python3-tkinter

# Arch
sudo pacman -S tk
```

## Need Help?

- Check the full [README.md](readme.md) for detailed documentation
- See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup
- Open an issue on GitHub for bugs or questions
