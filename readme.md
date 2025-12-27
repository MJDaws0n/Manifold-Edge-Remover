# **WARNING**
- Application may appear to hang, freeze, lag device, or do nothing, note it does take a while to perform tasks on complex models, so be patient.
- This is made partially using AI. That means there may be issues. Run at your own risk

# STL Manifold Edge Fixer

A cross-platform Python application that fixes non-manifold edges in STL files, making them suitable for 3D printing with Bambu Labs and other 3D printers.

> **🚀 New to this? Check out the [Quick Start Guide](QUICKSTART.md) for step-by-step instructions!**

## Features

- 🖥️ **Cross-platform**: Works on macOS, Windows, and Linux
- 🎯 **Easy to use**: Simple GUI with file selector
- 🔧 **Automatic fixing**: Fixes manifold edges, duplicate vertices, degenerate faces, and more
- 💾 **Safe**: Saves fixed files with `_FIXED` suffix, preserving the original
- 📦 **Self-contained**: Automatic dependency installation and virtual environment setup
- 🚀 **Ready for 3D printing**: Optimized for Bambu Labs and other 3D printers

## What Does It Fix?

The application automatically fixes common mesh issues:
- **Non-manifold edges**: Edges shared by more than two faces
- **Duplicate vertices**: Multiple vertices at the same position
- **Degenerate faces**: Faces with zero area
- **Duplicate faces**: Identical faces in the mesh
- **Holes**: Small holes in the mesh (when possible)
- **Normals**: Ensures face normals point outward consistently

## Quick Start

### macOS

1. **First time setup** (installs dependencies in a virtual environment):
   ```bash
   ./setup_mac.sh
   ```

2. **Run the application**:
   ```bash
   ./run_mac.sh
   ```

### Windows

1. **First time setup**:
   - Double-click `setup_windows.bat`
   - Or run in Command Prompt:
     ```cmd
     setup_windows.bat
     ```

2. **Run the application**:
   - Double-click `run_windows.bat`
   - Or run in Command Prompt:
     ```cmd
     run_windows.bat
     ```

### Linux

1. **First time setup**:
   ```bash
   ./setup_linux.sh
   ```

2. **Run the application**:
   ```bash
   ./run_linux.sh
   ```

## How to Use

1. Run the application using the appropriate script for your OS
2. A file selection dialog will appear
3. Select your STL file
4. The application will process the file and display progress in the console
5. The fixed file will be saved in the same directory as the original with `_FIXED` added before the extension
   - Example: `model.stl` → `model_FIXED.stl`
6. A success dialog will show you the results

## Building Executables

You can create standalone executables that don't require Python to be installed.

### Prerequisites

First, install PyInstaller in your virtual environment:

**macOS/Linux:**
```bash
source venv/bin/activate
pip install pyinstaller
```

**Windows:**
```cmd
venv\Scripts\activate.bat
pip install pyinstaller
```

### Build Instructions

#### macOS

```bash
source venv/bin/activate
pyinstaller --onefile --windowed --name STL_Fixer stl_fixer.py
```

The executable will be in the `dist/` folder: `dist/STL_Fixer`

To create a `.app` bundle:
```bash
pyinstaller --onedir --windowed --name STL_Fixer stl_fixer.py
```

#### Windows

```cmd
venv\Scripts\activate.bat
pyinstaller --onefile --windowed --name STL_Fixer.exe stl_fixer.py
```

The executable will be in the `dist\` folder: `dist\STL_Fixer.exe`

#### Linux

```bash
source venv/bin/activate
pyinstaller --onefile --windowed --name STL_Fixer stl_fixer.py
```

The executable will be in the `dist/` folder: `dist/STL_Fixer`

### PyInstaller Options Explained

- `--onefile`: Creates a single executable file
- `--windowed`: Hides the console window (GUI mode)
- `--name`: Sets the name of the executable
- `--icon`: (Optional) Sets a custom icon for the executable

## Requirements

- Python 3.7 or higher
- Dependencies (automatically installed by setup scripts):
  - trimesh >= 4.0.0
  - numpy >= 1.20.0
   - scipy >= 1.8.0
   - networkx >= 2.8
   - manifold3d >= 3.0.0
   - scikit-image >= 0.22.0

## Manual Installation

If you prefer to install manually:

1. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   ```

2. **Activate the virtual environment**:
   - macOS/Linux: `source venv/bin/activate`
   - Windows: `venv\Scripts\activate.bat`

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python stl_fixer.py
   ```

## Troubleshooting

### macOS

- **"Python 3 is not installed"**: Install Python from [python.org](https://www.python.org/downloads/) or use Homebrew: `brew install python3`
- **Permission denied**: Make scripts executable with `chmod +x setup_mac.sh run_mac.sh`

### Windows

- **"Python is not installed"**: Install Python from [python.org](https://www.python.org/downloads/) and make sure to check "Add Python to PATH" during installation
- **Scripts won't run**: Right-click the script and select "Run as administrator"

### Linux

- **"Python 3 is not installed"**: Install using your package manager:
  - Ubuntu/Debian: `sudo apt-get install python3 python3-venv python3-pip`
  - Fedora: `sudo dnf install python3 python3-pip`
  - Arch: `sudo pacman -S python python-pip`
- **Permission denied**: Make scripts executable with `chmod +x setup_linux.sh run_linux.sh`

### Common Issues

- **"No module named install" after running `python -m install ...`**: Use pip instead:
   - `python -m pip install -r requirements.txt`

- **Tkinter not found**: On Linux, you may need to install tkinter separately:
  - Ubuntu/Debian: `sudo apt-get install python3-tk`
  - Fedora: `sudo dnf install python3-tkinter`
  - Arch: `sudo pacman -S tk`

## License

This project is open source and available under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

Built with:
- [Trimesh](https://trimsh.org/) - Powerful mesh processing library
- [NumPy](https://numpy.org/) - Numerical computing library
- [SciPy](https://scipy.org/) - Scientific computing library
