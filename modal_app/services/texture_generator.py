"""Texture Generation Service for Hunyuan3D-2.1.

This service wraps the Hunyuan3DPaintPipeline for serverless GPU execution.
It runs on L40S GPU (48GB VRAM) and uses memory snapshots for fast cold starts.

Interface:
    Input: GLB mesh bytes + reference PNG/JPEG image bytes
    Output: Textured GLB mesh with PBR materials as bytes
    Expected VRAM: ~21GB
    Expected inference time: ~60s

Usage on Modal:
    result_bytes = texture_generator.generate.remote(mesh_bytes, image_bytes, seed=42)
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

from PIL import Image

# =============================================================================
# Configuration Constants
# =============================================================================

# Texture generation quality settings
# max_num_view: Number of views for multi-view diffusion (6-9, higher = better quality)
# resolution: Output texture resolution (512 or 768, higher = sharper but slower)
TEXTURE_MAX_NUM_VIEWS = 6
TEXTURE_RESOLUTION = 512


class TextureGeneratorService:
    """Texture generation service (local implementation for testing).

    This class contains the core logic for texture generation, separated from
    Modal-specific decorators to enable unit testing without GPU.

    The Modal-decorated version (TextureGenerator) wraps this class.
    """

    def __init__(self, pipeline=None) -> None:
        """Initialize the service with optional dependencies.

        Args:
            pipeline: Texture generation pipeline (injected by Modal wrapper)

        For unit testing, pass None and mock internal methods instead.
        """
        self._pipeline = pipeline

    def generate(
        self,
        mesh_bytes: bytes,
        image_bytes: bytes,
        seed: int | None = None,
    ) -> bytes:
        """Generate a textured 3D mesh from a mesh and reference image.

        Args:
            mesh_bytes: GLB mesh as bytes (from ShapeGenerator)
            image_bytes: PNG or JPEG reference image as bytes
            seed: Optional seed for reproducibility

        Returns:
            Textured GLB mesh file as bytes

        Raises:
            ValueError: If mesh_bytes or image_bytes is empty or invalid
            RuntimeError: If inference fails
        """
        if not mesh_bytes:
            raise ValueError("mesh_bytes cannot be empty")

        if not image_bytes:
            raise ValueError("image_bytes cannot be empty")

        self._validate_mesh(mesh_bytes)
        image_format = self._validate_and_get_image_format(image_bytes)

        # Use shared temp directory with automatic cleanup
        with tempfile.TemporaryDirectory(prefix="texture_gen_") as temp_dir:
            mesh_path = self._write_mesh_to_temp(mesh_bytes, temp_dir)
            image_path = self._write_image_to_temp(image_bytes, temp_dir, image_format)
            output_path = self._run_pipeline(
                mesh_path=mesh_path,
                image_path=image_path,
                seed=seed,
            )
            glb_bytes = self._read_output_glb(output_path)

        return glb_bytes

    def _validate_mesh(self, mesh_bytes: bytes) -> None:
        """Validate mesh bytes have valid GLB header.

        Args:
            mesh_bytes: GLB mesh as bytes

        Raises:
            ValueError: If bytes don't have valid GLB header
        """
        # GLB files start with magic bytes: 'glTF'
        if len(mesh_bytes) < 4 or mesh_bytes[:4] != b"glTF":
            raise ValueError("invalid mesh data: not a valid GLB file")

    def _validate_and_get_image_format(self, image_bytes: bytes) -> str:
        """Validate image bytes can be decoded and return the format.

        Args:
            image_bytes: PNG or JPEG image as bytes

        Returns:
            Image format string (e.g., "PNG", "JPEG")

        Raises:
            ValueError: If bytes cannot be decoded as an image
        """
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()  # Force decode to catch corrupt images
            return image.format or "PNG"  # Default to PNG if format unknown
        except Exception as e:
            raise ValueError(f"invalid image data: {e}") from e

    def _write_mesh_to_temp(self, mesh_bytes: bytes, temp_dir: str) -> str:
        """Write mesh bytes to a temporary file.

        Args:
            mesh_bytes: GLB mesh as bytes
            temp_dir: Directory to write the file to

        Returns:
            Path to the temporary mesh file
        """
        mesh_path = Path(temp_dir) / "input_mesh.glb"
        mesh_path.write_bytes(mesh_bytes)
        return str(mesh_path)

    def _write_image_to_temp(
        self, image_bytes: bytes, temp_dir: str, image_format: str
    ) -> str:
        """Write image bytes to a temporary file.

        Args:
            image_bytes: PNG or JPEG image as bytes
            temp_dir: Directory to write the file to
            image_format: Image format string (e.g., "PNG", "JPEG")

        Returns:
            Path to the temporary image file
        """
        ext = ".png" if image_format == "PNG" else ".jpg"
        image_path = Path(temp_dir) / f"input_image{ext}"
        image_path.write_bytes(image_bytes)
        return str(image_path)

    def _run_pipeline(
        self,
        *,
        mesh_path: str,
        image_path: str,
        seed: int | None = None,
    ) -> str:
        """Run the texture generation pipeline.

        Args:
            mesh_path: Path to input mesh file
            image_path: Path to reference image file
            seed: Optional seed for reproducibility

        Returns:
            Path to output textured GLB file
        """
        if self._pipeline is None:
            raise RuntimeError("Pipeline not initialized")

        # Set seed for reproducibility
        if seed is not None:
            import torch

            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Create output paths
        mesh_dir = Path(mesh_path).parent
        output_obj_path = str(mesh_dir / "textured_mesh.obj")
        output_glb_path = str(mesh_dir / "textured_mesh.glb")

        # Run pipeline - outputs OBJ + texture files (albedo, metallic, roughness)
        # Using save_glb=False to avoid bpy dependency, matching production path
        self._pipeline(
            mesh_path=mesh_path,
            image_path=image_path,
            output_mesh_path=output_obj_path,
            use_remesh=True,  # Remesh for better texture quality
            save_glb=False,  # Use pygltflib conversion instead of bpy
        )

        # Convert OBJ to GLB with PBR materials using pygltflib
        # This matches the production code path in gradio_app.py/model_worker.py
        self._convert_obj_to_glb_with_pbr(output_obj_path, output_glb_path)

        return output_glb_path

    def _convert_obj_to_glb_with_pbr(self, obj_path: str, glb_path: str) -> None:
        """Convert OBJ with textures to GLB with PBR materials.

        Uses pygltflib to create GLB with proper PBR material setup,
        matching the original production code in hy3dpaint/convert_utils.py.

        Args:
            obj_path: Path to OBJ file with associated texture files
            glb_path: Output path for GLB file
        """
        import os

        # Import conversion function from original codebase
        from hy3dpaint.convert_utils import create_glb_with_pbr_materials

        # Build texture paths using original naming convention
        textures_dict = {
            "albedo": obj_path.replace(".obj", ".jpg"),
            "metallic": obj_path.replace(".obj", "_metallic.jpg"),
            "roughness": obj_path.replace(".obj", "_roughness.jpg"),
        }

        # The conversion function writes temp files to current directory
        # Change to output directory to keep temp files contained
        original_dir = os.getcwd()
        try:
            os.chdir(Path(obj_path).parent)
            create_glb_with_pbr_materials(obj_path, textures_dict, glb_path)
        finally:
            os.chdir(original_dir)

    def _read_output_glb(self, output_path: str) -> bytes:
        """Read output GLB file as bytes.

        Args:
            output_path: Path to the output GLB file

        Returns:
            GLB file as bytes
        """
        return Path(output_path).read_bytes()


# =============================================================================
# Modal Service Class
# =============================================================================

import modal

from modal_app.image import (
    APP_DIR,
    DIFFERENTIABLE_RENDERER_PATH,
    DINOV2_REPO_ID,
    HUNYUAN3D_REPO_ID,
    REALESRGAN_CKPT_PATH,
    hunyuan_image,
)

# Create image with modal_app mounted for imports
texture_generator_image = hunyuan_image.add_local_python_source("modal_app")

app = modal.App("hunyuan3d-texture-generator")


@app.cls(
    image=texture_generator_image,
    gpu="L40S",  # 48GB VRAM for texture generation
    timeout=600,
    scaledown_window=300,
    enable_memory_snapshot=True,
)
class TextureGenerator:
    """Modal-deployed texture generation service.

    Uses memory snapshots for fast cold starts:
    - snap=True: Loads models to CPU during image build (~40s)
    - snap=False: Moves models to GPU after restore (~10s)
    - Total cold start after snapshot: ~15-20s (vs ~60s without)
    """

    @modal.enter(snap=True)
    def load_models_to_cpu(self):
        """Load models to CPU memory (runs during image build for snapshotting)."""
        import os
        import sys
        from types import ModuleType

        # Set up Python paths for hy3dpaint imports
        sys.path.insert(0, f"{APP_DIR}/hy3dpaint")
        sys.path.insert(0, DIFFERENTIABLE_RENDERER_PATH)
        os.chdir(APP_DIR)

        # Mock bpy module to satisfy import - we use pygltflib path, not bpy
        # The upstream textureGenPipeline imports mesh_utils which has 'import bpy'
        # at module level. Since we use save_glb=False + pygltflib conversion,
        # bpy is never actually called, but we need the import to succeed.
        if "bpy" not in sys.modules:
            mock_bpy = ModuleType("bpy")
            sys.modules["bpy"] = mock_bpy

        # Import pipeline and config
        from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

        # Configure pipeline with Modal container paths
        config = Hunyuan3DPaintConfig(TEXTURE_MAX_NUM_VIEWS, TEXTURE_RESOLUTION)

        # Override paths for container environment
        config.multiview_cfg_path = f"{APP_DIR}/hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
        config.custom_pipeline = f"{APP_DIR}/hy3dpaint/hunyuanpaintpbr"
        # Use HF repo IDs so snapshot_download()/from_pretrained() find models in cache
        config.multiview_pretrained_path = HUNYUAN3D_REPO_ID
        config.dino_ckpt_path = DINOV2_REPO_ID
        config.realesrgan_ckpt_path = REALESRGAN_CKPT_PATH

        # Load pipeline (this loads models to CPU initially)
        self._pipeline = Hunyuan3DPaintPipeline(config)

        # Initialize service with dependencies via constructor injection
        self._service = TextureGeneratorService(pipeline=self._pipeline)

    @modal.enter(snap=False)
    def move_models_to_gpu(self):
        """Move models to GPU after restore from snapshot."""
        # The pipeline internally handles moving models to GPU when called
        # If the pipeline has explicit GPU methods, call them here
        pass

    @modal.method()
    def generate(
        self,
        mesh_bytes: bytes,
        image_bytes: bytes,
        seed: int | None = None,
    ) -> bytes:
        """Generate a textured 3D mesh from a mesh and reference image.

        Args:
            mesh_bytes: GLB mesh as bytes (from ShapeGenerator)
            image_bytes: PNG or JPEG reference image as bytes
            seed: Optional seed for reproducibility

        Returns:
            Textured GLB mesh file as bytes
        """
        return self._service.generate(mesh_bytes, image_bytes, seed=seed)
