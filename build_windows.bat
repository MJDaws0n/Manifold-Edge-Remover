@echo off
REM Build script for creating a standalone Windows executable

echo ================================================
echo Building STL Fixer for Windows
echo ================================================
echo.

REM Get the directory where this script is located
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist "venv\" (
    echo Virtual environment not found. Running setup...
    call setup_windows.bat
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install PyInstaller if not already installed
echo Installing PyInstaller...
pip install pyinstaller

echo.

REM Clean previous builds
if exist "build\" (
    echo Cleaning previous build...
    rmdir /s /q build
)

if exist "dist\" (
    echo Cleaning previous dist...
    rmdir /s /q dist
)

echo.

REM Build the executable
echo Building executable...
pyinstaller --onefile --windowed --name STL_Fixer.exe stl_fixer.py

echo.
echo ================================================
echo [32m✓ Build complete![0m
echo ================================================
echo.
echo The standalone application is located at:
echo   dist\STL_Fixer.exe
echo.
echo You can move this file anywhere and run it without Python installed.
echo.
pause
