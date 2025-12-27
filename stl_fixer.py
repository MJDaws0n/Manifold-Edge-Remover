#!/usr/bin/env python3
"""
STL Manifold Edge Fixer
A cross-platform GUI application to fix non-manifold edges in STL files for 3D printing.
"""

import sys
import os
from pathlib import Path


def _dependency_error_message(missing: list[str]) -> str:
    joined = ", ".join(missing)
    base = (
        "Missing Python dependencies: " + joined + "\n\n"
        "Install with:\n"
        "  python -m pip install -r requirements.txt\n\n"
        "If you're using the repo embedded env, run:\n"
        "  ./app/bin/python -m pip install -r requirements.txt\n"
    )
    return base


def _require_imports():
    missing: list[str] = []

    try:
        import tkinter as tk  # noqa: F401
        from tkinter import filedialog, messagebox  # noqa: F401
    except Exception:
        # tkinter is optional for some environments; fail later with a messagebox fallback.
        missing.append("tkinter")

    try:
        import trimesh  # noqa: F401
    except ModuleNotFoundError:
        missing.append("trimesh")

    try:
        import numpy as np  # noqa: F401
    except ModuleNotFoundError:
        missing.append("numpy")

    # Optional but strongly recommended for best results with trimesh operations
    try:
        import scipy  # noqa: F401
    except ModuleNotFoundError:
        missing.append("scipy")

    try:
        import networkx  # noqa: F401
    except ModuleNotFoundError:
        missing.append("networkx")

    try:
        import manifold3d  # noqa: F401
    except ModuleNotFoundError:
        missing.append("manifold3d")

    if missing:
        msg = _dependency_error_message(missing)
        print(msg)
        # Try to show a GUI error if tkinter is available
        try:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Missing dependencies", msg)
        except Exception:
            pass
        raise SystemExit(1)


_require_imports()

import tkinter as tk
from tkinter import filedialog, messagebox
import trimesh
import numpy as np


def _edge_issue_counts(mesh: "trimesh.Trimesh") -> tuple[int, int]:
    """Return (boundary_edges, nonmanifold_edges) for the current faces."""
    from trimesh.geometry import faces_to_edges

    if len(mesh.faces) == 0:
        return (0, 0)

    edges = faces_to_edges(mesh.faces)
    edges = np.sort(edges, axis=1)
    edges = np.ascontiguousarray(edges)
    edge_view = edges.view([("a", edges.dtype), ("b", edges.dtype)])
    _, counts = np.unique(edge_view, return_counts=True)
    # counts corresponds to unique edges; to classify we just need how many are 1 or >2
    boundary = int(np.sum(counts == 1))
    nonmanifold = int(np.sum(counts > 2))
    return boundary, nonmanifold


def _manifoldize_with_manifold3d(mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
    """Attempt to convert arbitrary triangle soup into a manifold surface using manifold3d."""
    from manifold3d import Manifold, Mesh

    # manifold3d expects float32 vertices and uint32 faces
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)

    m = Manifold(mesh=Mesh(vert_properties=verts, tri_verts=faces))
    if getattr(m, "is_empty", lambda: False)():
        raise ValueError("manifold3d produced an empty result")

    out = m.to_mesh()
    return trimesh.Trimesh(vertices=out.vert_properties, faces=out.tri_verts, process=False)


def _cap_boundary_loops_with_manifold3d(mesh: "trimesh.Trimesh") -> int:
    """Cap open boundaries by triangulating each boundary loop.

    This targets slicers (like Bambu Studio) that flag boundary/open edges as
    non-manifold. Works best when boundary loops are reasonably planar.

    Returns
    -------
    int
      Number of faces added.
    """

    import manifold3d
    import networkx as nx
    from trimesh.geometry import faces_to_edges

    if len(mesh.faces) == 0:
        return 0

    faces = np.asarray(mesh.faces)
    verts = np.asarray(mesh.vertices)

    # Find boundary edges (edges used exactly once)
    edges_dir = faces_to_edges(faces)
    edges_und = np.sort(edges_dir, axis=1)
    edges_und = np.ascontiguousarray(edges_und)
    edge_view = edges_und.view([("a", edges_und.dtype), ("b", edges_und.dtype)])
    _, inverse, counts = np.unique(edge_view, return_inverse=True, return_counts=True)
    boundary_mask = (counts[inverse] == 1).reshape(-1)
    boundary_edges = edges_und[np.nonzero(boundary_mask)[0]]
    if boundary_edges.size == 0:
        return 0

    # Use cycle basis on boundary graph (robust and fast for our small boundary sets)
    g = nx.from_edgelist(boundary_edges)
    loops = nx.cycle_basis(g)
    if not loops:
        return 0

    new_faces_all: list[np.ndarray] = []
    faces_added = 0

    for loop in loops:
        # project boundary loop to best-fit plane (PCA)
        pts = verts[np.array(loop, dtype=int)]
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        # if points are nearly colinear, skip
        _, s, vh = np.linalg.svd(centered, full_matrices=False)
        if s.size < 2 or s[-1] < 1e-12:
            continue
        # basis vectors spanning plane
        u = vh[0]
        v = vh[1]
        ring2d = np.column_stack((centered @ u, centered @ v)).astype(np.float64)

        try:
            tri = manifold3d.triangulate([ring2d]).astype(np.int64)
        except Exception:
            continue

        if tri.size == 0:
            continue

        # map triangulation indices back to original vertex indices
        loop_idx = np.array(loop, dtype=np.int64)
        cap_faces = loop_idx[tri]

        new_faces_all.append(cap_faces)
        faces_added += int(len(cap_faces))

    if faces_added == 0:
        return 0

    mesh.faces = np.vstack((mesh.faces, np.vstack(new_faces_all)))
    mesh._cache.verify()
    return faces_added


def _stitch_boundaries(mesh: "trimesh.Trimesh") -> tuple[int, int]:
    """Cap boundary loops with triangle fans (fast, preserves overall surface).

    Uses `trimesh.repair.stitch` which builds triangle fans over boundary curves.
    Returns (faces_added, verts_added).
    """

    from trimesh.repair import stitch

    fan, verts_new = stitch(mesh, faces=None, insert_vertices=True)
    faces_added = int(len(fan)) if hasattr(fan, "__len__") else 0
    verts_added = int(len(verts_new)) if hasattr(verts_new, "__len__") else 0

    if verts_added:
        mesh.vertices = np.vstack((mesh.vertices, verts_new))
    if faces_added:
        mesh.faces = np.vstack((mesh.faces, fan))
        mesh._cache.verify()

    return faces_added, verts_added


def _voxel_watertight_remesh(
    mesh: "trimesh.Trimesh",
    *,
    original_bounds: np.ndarray,
    original_extents: np.ndarray,
) -> "trimesh.Trimesh":
    """Remesh via voxelization + marching cubes to force watertight output.

    This is a last-resort fallback: it will modify geometry slightly but is
    effective at eliminating open boundary edges that slicers flag as
    non-manifold.
    """

    ext = np.asarray(mesh.extents, dtype=float)
    max_dim = float(np.max(ext)) if ext.size else 0.0
    if not np.isfinite(max_dim) or max_dim <= 0.0:
        raise ValueError("Invalid mesh extents for voxel remesh")

    # Pick voxel pitch: keep it bounded so it doesn't freeze machines.
    # Earlier "too-fine" pitch caused massive slowdowns.
    target_resolution = 450.0  # voxels along max axis
    pitch = max_dim / target_resolution
    pitch = float(np.clip(pitch, 0.2, 1.0))

    # If the model would exceed our max voxel budget, increase pitch.
    # This caps runtime/memory on large meshes.
    max_axis_voxels = 520.0
    if max_dim / pitch > max_axis_voxels:
        pitch = max_dim / max_axis_voxels

    vg = mesh.voxelized(pitch)
    vg = vg.fill()
    out = vg.marching_cubes
    # marching cubes returns a new trimesh already
    out.remove_degenerate_faces()
    out.merge_vertices()

    # Light smoothing to reduce voxel stair-stepping.
    # (smoothing can slightly shrink/grow; we'll correct scale below)
    try:
        from trimesh.smoothing import filter_taubin

        # Taubin smoothing reduces voxel stair-stepping with less shrinkage
        filter_taubin(out, lamb=0.5, nu=-0.53, iterations=10)
    except Exception:
        # smoothing is best-effort
        pass

    # Preserve scale: re-scale and re-center to match the original bounds.
    try:
        new_extents = np.asarray(out.extents, dtype=float)
        denom = float(np.max(new_extents))
        numer = float(np.max(np.asarray(original_extents, dtype=float)))
        if np.isfinite(denom) and denom > 0 and np.isfinite(numer) and numer > 0:
            out.apply_scale(numer / denom)
    except Exception:
        pass

    try:
        orig_center = np.asarray(original_bounds, dtype=float).mean(axis=0)
        new_center = np.asarray(out.bounds, dtype=float).mean(axis=0)
        out.apply_translation(orig_center - new_center)
    except Exception:
        pass

    return out


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
        original_bounds = np.asarray(mesh.bounds, dtype=float)
        original_extents = np.asarray(mesh.extents, dtype=float)
        b0, n0 = _edge_issue_counts(mesh)
        if b0 or n0:
            print(f"Edge issues (pre-fix): {b0} boundary edges, {n0} non-manifold edges")
        
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
        
        # Fill/cap holes if present
        if not mesh.is_watertight:
            try:
                if mesh.fill_holes():
                    print("Filled some small holes in mesh")
            except Exception as e:
                print(f"Warning: Small hole filling failed: {e}")

            # Cap larger boundary loops (addresses slicer 'non-manifold edges')
            # Run a few passes as capping can expose additional loops.
            last_boundary = None
            for _ in range(5):
                boundary, nonmanifold = _edge_issue_counts(mesh)
                if boundary == 0 and nonmanifold == 0:
                    break
                if last_boundary is not None and boundary >= last_boundary:
                    break
                last_boundary = boundary
                if boundary == 0:
                    break
                try:
                    added = _cap_boundary_loops_with_manifold3d(mesh)
                    if added:
                        print(f"Capped boundary loops (+{added} faces)")
                        # cleanup after new faces
                        mesh.merge_vertices()
                        mesh.remove_unreferenced_vertices()
                    else:
                        break
                except Exception as e:
                    print(f"Warning: Boundary capping failed: {e}")
                    break
        
        # Robust manifold repair using manifold3d (preferred) to address Bambu Studio non-manifold edges
        # This step can change topology but is usually what slicers expect.
        b1, n1 = _edge_issue_counts(mesh)
        if b1 or n1 or not mesh.is_watertight:
            try:
                mesh = _manifoldize_with_manifold3d(mesh)
                print("Applied manifold3d repair")
            except Exception as e:
                print(f"Warning: manifold3d repair failed: {e}")

        # If boundaries still remain, stitch them with triangle fans.
        # This is much faster and less destructive than voxel remesh.
        b_st, n_st = _edge_issue_counts(mesh)
        if b_st or n_st or not mesh.is_watertight:
            try:
                faces_added, verts_added = _stitch_boundaries(mesh)
                if faces_added or verts_added:
                    print(f"Stitched boundaries (+{faces_added} faces, +{verts_added} verts)")
                    mesh.merge_vertices()
                    mesh.remove_unreferenced_vertices()
            except Exception as e:
                print(f"Warning: boundary stitch failed: {e}")

        # If we still have open boundaries, use voxel remesh as a last resort.
        b_vox, n_vox = _edge_issue_counts(mesh)
        if b_vox or n_vox or not mesh.is_watertight:
            if b_vox:
                print("Open boundaries remain; applying voxel watertight remesh (last resort)")
            try:
                # scikit-image provides skimage (needed for marching cubes)
                try:
                    import skimage  # noqa: F401
                except ModuleNotFoundError:
                    raise ModuleNotFoundError(
                        "No module named 'skimage' (install scikit-image to enable voxel watertight remesh)"
                    )
                mesh = _voxel_watertight_remesh(
                    mesh,
                    original_bounds=original_bounds,
                    original_extents=original_extents,
                )
                print("Applied voxel watertight remesh")
            except Exception as e:
                print(f"Warning: voxel watertight remesh failed: {e}")

        # Fix normals to ensure they point outward
        try:
            # Avoid trimesh auto-detecting multibody (can require scipy for connected components)
            mesh.fix_normals(multibody=False)
        except ModuleNotFoundError as e:
            print(f"Warning: Could not fix normals due to missing dependency: {e}")
        
        # Final cleanup
        mesh.merge_vertices()
        mesh.remove_unreferenced_vertices()

        b2, n2 = _edge_issue_counts(mesh)
        if b2 or n2:
            print(f"Edge issues (post-fix): {b2} boundary edges, {n2} non-manifold edges")
        
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
