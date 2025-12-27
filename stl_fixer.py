#!/usr/bin/env python3
"""
STL Manifold Edge Fixer
A cross-platform GUI application to fix non-manifold edges in STL files for 3D printing.
"""

import sys
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import trimesh
import numpy as np


class STLFixerApp:
    """Main application class for STL manifold edge fixing."""
    
    def __init__(self):
        """Initialize the application."""
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the main window
        
    def select_file(self):
        """Open a file dialog to select an STL file."""
        file_path = filedialog.askopenfilename(
            title="Select STL file to fix",
            filetypes=[
                ("STL files", "*.stl"),
                ("All files", "*.*")
            ]
        )
        return file_path
    
    def fix_manifold_edges(self, mesh):
        """
        Fix non-manifold edges in the mesh.
        
        Args:
            mesh: A trimesh.Trimesh object
            
        Returns:
            A fixed trimesh.Trimesh object
        """
        print("Analyzing mesh...")
        print(f"Original mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Check if mesh is watertight
        if mesh.is_watertight:
            print("Mesh is already watertight!")
        else:
            print("Mesh has manifold issues, attempting to fix...")
        
        # Remove duplicate vertices
        mesh.merge_vertices()
        
        # Remove degenerate faces (faces with zero area)
        # Keep only non-degenerate faces
        valid_faces = mesh.nondegenerate_faces()
        if len(valid_faces) < len(mesh.faces):
            removed_count = len(mesh.faces) - len(valid_faces)
            mesh.update_faces(valid_faces)
            print(f"Removed {removed_count} degenerate faces")
        
        # Remove duplicate faces - keep only unique faces
        unique_faces = mesh.unique_faces()
        if len(unique_faces) < len(mesh.faces):
            removed_count = len(mesh.faces) - len(unique_faces)
            mesh.update_faces(unique_faces)
            print(f"Removed {removed_count} duplicate faces")
        
        # Remove infinite values
        mesh.remove_infinite_values()
        
        # Fill holes if present
        if not mesh.is_watertight:
            try:
                # Try to fill small holes
                mesh.fill_holes()
                print("Filled holes in mesh")
            except Exception as e:
                print(f"Warning: Could not fill all holes: {e}")
        
        # Fix normals to ensure they point outward
        mesh.fix_normals()
        
        # Final cleanup
        mesh.merge_vertices()
        mesh.remove_unreferenced_vertices()
        
        print(f"Fixed mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Check if mesh is now watertight
        if mesh.is_watertight:
            print("✓ Mesh is now watertight and ready for 3D printing!")
        else:
            print("⚠ Mesh still has some issues, but has been improved for 3D printing.")
        
        return mesh
    
    def save_fixed_file(self, mesh, original_path):
        """
        Save the fixed mesh with _FIXED suffix.
        
        Args:
            mesh: A trimesh.Trimesh object
            original_path: Path to the original file
            
        Returns:
            Path to the saved file
        """
        path = Path(original_path)
        new_filename = f"{path.stem}_FIXED{path.suffix}"
        output_path = path.parent / new_filename
        
        # Export the fixed mesh
        mesh.export(str(output_path))
        
        return str(output_path)
    
    def run(self):
        """Main application logic."""
        print("=" * 60)
        print("STL Manifold Edge Fixer")
        print("=" * 60)
        print()
        
        # Select input file
        input_file = self.select_file()
        
        if not input_file:
            print("No file selected. Exiting.")
            return
        
        print(f"Selected file: {input_file}")
        print()
        
        try:
            # Load the mesh
            print("Loading STL file...")
            mesh = trimesh.load(input_file)
            
            # Handle Scene objects (multiple meshes)
            if isinstance(mesh, trimesh.Scene):
                print("File contains multiple meshes, combining them...")
                mesh = trimesh.util.concatenate(
                    [geom for geom in mesh.geometry.values() 
                     if isinstance(geom, trimesh.Trimesh)]
                )
            
            print()
            
            # Fix the mesh
            fixed_mesh = self.fix_manifold_edges(mesh)
            
            print()
            
            # Save the fixed file
            print("Saving fixed file...")
            output_file = self.save_fixed_file(fixed_mesh, input_file)
            
            print(f"✓ Fixed file saved to: {output_file}")
            print()
            
            # Show success message
            messagebox.showinfo(
                "Success",
                f"STL file has been fixed and saved to:\n\n{output_file}\n\n"
                f"Original: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces\n"
                f"Fixed: {len(fixed_mesh.vertices)} vertices, {len(fixed_mesh.faces)} faces"
            )
            
        except Exception as e:
            error_msg = f"Error processing file: {str(e)}"
            print(f"✗ {error_msg}")
            messagebox.showerror("Error", error_msg)
            raise


def main():
    """Entry point for the application."""
    app = STLFixerApp()
    app.run()


if __name__ == "__main__":
    main()
