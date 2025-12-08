#!/usr/bin/env python3
"""Local mesh post-processing utilities.

These utilities run on the client side (no GPU required) and provide
additional mesh processing after downloading from the Modal API:
- Format conversion (GLB -> OBJ/PLY/STL)
- Face reduction for smaller file sizes

Usage:
    from mesh_utils import convert_format, reduce_faces

    # Convert GLB to OBJ
    obj_bytes = convert_format(glb_bytes, "obj")

    # Reduce mesh complexity
    simplified_bytes = reduce_faces(glb_bytes, target_faces=10000)

Requirements:
    pip install trimesh numpy
"""

from __future__ import annotations

import io
from typing import Literal

import trimesh

# =============================================================================
# Format Conversion
# =============================================================================

FormatType = Literal["obj", "ply", "stl", "glb", "gltf"]

SUPPORTED_FORMATS = {"obj", "ply", "stl", "glb", "gltf"}


def convert_format(glb_bytes: bytes, output_format: FormatType) -> bytes:
    """Convert a GLB mesh to another format.

    Args:
        glb_bytes: Input GLB mesh as bytes
        output_format: Target format ("obj", "ply", "stl", "glb", "gltf")

    Returns:
        Converted mesh as bytes

    Raises:
        ValueError: If format is not supported or input is invalid

    Note:
        Some formats (OBJ, STL, PLY) may lose texture/material information.
        GLB and glTF preserve PBR materials.
    """
    if output_format not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format: {output_format}. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    if not glb_bytes or len(glb_bytes) < 4:
        raise ValueError("Input bytes are empty or too small")

    # Load mesh from GLB bytes
    mesh = trimesh.load(io.BytesIO(glb_bytes), file_type="glb")

    # Handle Scene objects (GLB can contain multiple meshes)
    if isinstance(mesh, trimesh.Scene):
        # Combine all geometries into a single mesh
        combined = trimesh.util.concatenate(
            [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )
        mesh = combined

    # Export to target format
    buffer = io.BytesIO()

    if output_format == "glb":
        mesh.export(buffer, file_type="glb")
    elif output_format == "gltf":
        mesh.export(buffer, file_type="gltf")
    elif output_format == "obj":
        mesh.export(buffer, file_type="obj")
    elif output_format == "ply":
        mesh.export(buffer, file_type="ply")
    elif output_format == "stl":
        mesh.export(buffer, file_type="stl")

    buffer.seek(0)
    return buffer.read()


# =============================================================================
# Face Reduction
# =============================================================================


def reduce_faces(
    glb_bytes: bytes,
    target_faces: int = 10000,
    preserve_textures: bool = True,
) -> bytes:
    """Reduce the number of faces in a mesh while preserving shape.

    Uses quadric decimation to simplify the mesh. Useful for:
    - Reducing file size for web viewing
    - Preparing meshes for 3D printing
    - Creating LOD (level of detail) versions

    Args:
        glb_bytes: Input GLB mesh as bytes
        target_faces: Target number of faces (approximate)
        preserve_textures: Reserved for future use (currently ignored)

    Returns:
        Simplified GLB mesh as bytes

    Raises:
        ValueError: If input is invalid
        AttributeError: If simplification backend (open3d) is not available

    Note:
        Requires trimesh with open3d backend for quadric decimation.
        The actual face count may differ slightly from target_faces.
        Very aggressive reduction may cause visual artifacts.
    """
    if target_faces < 4:
        raise ValueError("target_faces must be at least 4 (tetrahedron)")

    if not glb_bytes or len(glb_bytes) < 4:
        raise ValueError("Input bytes are empty or too small")

    # Load mesh from GLB bytes
    mesh = trimesh.load(io.BytesIO(glb_bytes), file_type="glb")

    # Handle Scene objects
    if isinstance(mesh, trimesh.Scene):
        combined = trimesh.util.concatenate(
            [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )
        mesh = combined

    original_faces = len(mesh.faces)

    if original_faces <= target_faces:
        # Already below target, return as-is
        buffer = io.BytesIO()
        mesh.export(buffer, file_type="glb")
        buffer.seek(0)
        return buffer.read()

    # Simplify using quadric decimation (requires open3d backend)
    simplified = mesh.simplify_quadric_decimation(target_faces)

    # Export result
    buffer = io.BytesIO()
    simplified.export(buffer, file_type="glb")
    buffer.seek(0)
    return buffer.read()


# =============================================================================
# Mesh Info
# =============================================================================


def get_mesh_info(glb_bytes: bytes) -> dict:
    """Get information about a mesh.

    Args:
        glb_bytes: Input GLB mesh as bytes

    Returns:
        Dictionary with mesh information:
        - vertex_count: Number of vertices
        - face_count: Number of faces
        - has_vertex_colors: Whether mesh has vertex colors
        - has_vertex_normals: Whether mesh has vertex normals
        - has_uv_coordinates: Whether mesh has UV coordinates
        - bounding_box: (min, max) coordinates
        - file_size_bytes: Size of input data

    Raises:
        ValueError: If input is invalid
    """
    if not glb_bytes or len(glb_bytes) < 4:
        raise ValueError("Input bytes are empty or too small")

    # Load mesh
    mesh = trimesh.load(io.BytesIO(glb_bytes), file_type="glb")

    # Handle Scene objects
    if isinstance(mesh, trimesh.Scene):
        combined = trimesh.util.concatenate(
            [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )
        mesh = combined

    # Gather info
    bounds = mesh.bounds
    info = {
        "vertex_count": len(mesh.vertices),
        "face_count": len(mesh.faces),
        "has_vertex_colors": mesh.visual is not None
        and hasattr(mesh.visual, "vertex_colors")
        and mesh.visual.vertex_colors is not None,
        "has_vertex_normals": mesh.vertex_normals is not None
        and len(mesh.vertex_normals) > 0,
        "has_uv_coordinates": mesh.visual is not None
        and hasattr(mesh.visual, "uv")
        and mesh.visual.uv is not None,
        "bounding_box": {
            "min": bounds[0].tolist(),
            "max": bounds[1].tolist(),
        },
        "file_size_bytes": len(glb_bytes),
    }

    return info


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    """Command-line interface for mesh utilities."""
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Mesh post-processing utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert GLB to OBJ
    python mesh_utils.py convert input.glb output.obj

    # Reduce mesh to 10k faces
    python mesh_utils.py reduce input.glb output.glb --faces 10000

    # Get mesh info
    python mesh_utils.py info input.glb
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert mesh format")
    convert_parser.add_argument("input", type=Path, help="Input GLB file")
    convert_parser.add_argument("output", type=Path, help="Output file")
    convert_parser.add_argument(
        "--format",
        choices=list(SUPPORTED_FORMATS),
        help="Output format (inferred from extension if not specified)",
    )

    # Reduce command
    reduce_parser = subparsers.add_parser("reduce", help="Reduce mesh face count")
    reduce_parser.add_argument("input", type=Path, help="Input GLB file")
    reduce_parser.add_argument("output", type=Path, help="Output GLB file")
    reduce_parser.add_argument(
        "--faces",
        type=int,
        default=10000,
        help="Target face count (default: 10000)",
    )

    # Info command
    info_parser = subparsers.add_parser("info", help="Show mesh information")
    info_parser.add_argument("input", type=Path, help="Input GLB file")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "convert":
        if not args.input.exists():
            print(f"Error: File not found: {args.input}")
            sys.exit(1)

        output_format = args.format
        if not output_format:
            output_format = args.output.suffix.lstrip(".")
            if output_format not in SUPPORTED_FORMATS:
                print(f"Error: Unknown format from extension: {output_format}")
                sys.exit(1)

        glb_bytes = args.input.read_bytes()
        result = convert_format(glb_bytes, output_format)
        args.output.write_bytes(result)
        print(f"Converted to {output_format}: {args.output}")

    elif args.command == "reduce":
        if not args.input.exists():
            print(f"Error: File not found: {args.input}")
            sys.exit(1)

        glb_bytes = args.input.read_bytes()
        info_before = get_mesh_info(glb_bytes)

        result = reduce_faces(glb_bytes, target_faces=args.faces)
        args.output.write_bytes(result)

        info_after = get_mesh_info(result)
        print(
            f"Reduced faces: {info_before['face_count']} -> {info_after['face_count']}"
        )
        print(
            f"File size: {info_before['file_size_bytes']:,} -> {info_after['file_size_bytes']:,} bytes"
        )
        print(f"Output: {args.output}")

    elif args.command == "info":
        if not args.input.exists():
            print(f"Error: File not found: {args.input}")
            sys.exit(1)

        glb_bytes = args.input.read_bytes()
        info = get_mesh_info(glb_bytes)

        print(f"Mesh Information: {args.input}")
        print(f"  Vertices: {info['vertex_count']:,}")
        print(f"  Faces: {info['face_count']:,}")
        print(f"  Has vertex colors: {info['has_vertex_colors']}")
        print(f"  Has vertex normals: {info['has_vertex_normals']}")
        print(f"  Has UV coordinates: {info['has_uv_coordinates']}")
        print(f"  Bounding box: {info['bounding_box']}")
        print(f"  File size: {info['file_size_bytes']:,} bytes")


if __name__ == "__main__":
    main()
