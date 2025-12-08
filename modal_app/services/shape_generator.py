"""Shape Generation Service for Hunyuan3D-2.1.

This service wraps the Hunyuan3DDiTFlowMatchingPipeline for serverless GPU execution.
It runs on A10G GPU (24GB VRAM) and uses memory snapshots for fast cold starts.

Interface:
    Input: PNG/JPEG image as bytes
    Output: GLB mesh as bytes
    Expected VRAM: ~10GB
    Expected inference time: ~30s

Usage on Modal:
    result_bytes = shape_generator.generate.remote(image_bytes, seed=42)
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    import trimesh


class ShapeGeneratorService:
    """Shape generation service (local implementation for testing).

    This class contains the core logic for shape generation, separated from
    Modal-specific decorators to enable unit testing without GPU.

    The Modal-decorated version (ShapeGenerator) wraps this class.
    """

    def __init__(self, pipeline=None, rembg=None) -> None:
        """Initialize the service with optional dependencies.

        Args:
            pipeline: Shape generation pipeline (injected by Modal wrapper)
            rembg: Background remover callable (injected by Modal wrapper)

        For unit testing, pass None and mock internal methods instead.
        """
        self._pipeline = pipeline
        self._rembg = rembg

    def generate(
        self,
        image_bytes: bytes,
        seed: int | None = None,
        steps: int = 30,
        guidance_scale: float = 5.0,
        octree_resolution: int = 256,
    ) -> bytes:
        """Generate a 3D mesh from an input image.

        Args:
            image_bytes: PNG or JPEG image as bytes
            seed: Optional seed for reproducibility
            steps: Number of inference steps (default: 30, higher = better quality)
            guidance_scale: Classifier-free guidance scale (default: 5.0)
            octree_resolution: Mesh octree resolution (default: 256, higher = more detail)

        Returns:
            GLB mesh file as bytes

        Raises:
            ValueError: If image_bytes is empty or invalid
            RuntimeError: If inference fails
        """
        if not image_bytes:
            raise ValueError("image_bytes cannot be empty")

        image = self._load_image(image_bytes)
        processed_image = self._remove_background(image)
        mesh = self._run_pipeline(
            image=processed_image,
            seed=seed,
            steps=steps,
            guidance_scale=guidance_scale,
            octree_resolution=octree_resolution,
        )
        glb_bytes = self._export_mesh(mesh)

        return glb_bytes

    def _load_image(self, image_bytes: bytes) -> Image.Image:
        """Load image bytes into PIL Image.

        Args:
            image_bytes: PNG or JPEG image as bytes

        Returns:
            PIL Image object

        Raises:
            ValueError: If bytes cannot be decoded as an image
        """
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()  # Force decode to catch corrupt images
            return image
        except Exception as e:
            raise ValueError(f"invalid image data: {e}") from e

    def _remove_background(self, image: Image.Image) -> Image.Image:
        """Remove background from image using rembg.

        Args:
            image: PIL Image (RGB or RGBA)

        Returns:
            PIL Image with transparent background (RGBA)
        """
        if self._rembg is None:
            raise RuntimeError("BackgroundRemover not initialized")

        return self._rembg(image)

    def _run_pipeline(
        self,
        *,
        image: Image.Image,
        seed: int | None = None,
        steps: int = 30,
        guidance_scale: float = 5.0,
        octree_resolution: int = 256,
    ) -> "trimesh.Trimesh":
        """Run the shape generation pipeline.

        Args:
            image: PIL Image with transparent background
            seed: Optional seed for reproducibility
            steps: Number of inference steps
            guidance_scale: Classifier-free guidance scale
            octree_resolution: Mesh octree resolution

        Returns:
            Generated trimesh object
        """
        if self._pipeline is None:
            raise RuntimeError("Pipeline not initialized")

        import torch

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)

        meshes = self._pipeline(
            image=image,
            generator=generator,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            octree_resolution=octree_resolution,
        )

        return meshes[0]

    def _export_mesh(self, mesh: "trimesh.Trimesh") -> bytes:
        """Export trimesh to GLB bytes.

        Args:
            mesh: Trimesh object to export

        Returns:
            GLB file as bytes
        """
        buffer = io.BytesIO()
        mesh.export(buffer, file_type="glb")
        buffer.seek(0)
        return buffer.read()


# =============================================================================
# Modal Service Class
# =============================================================================

import modal

from modal_app.image import (
    APP_DIR,
    HUNYUAN3D_REPO_ID,
    HY3DSHAPE_PACKAGE_PATH,
    hunyuan_image,
)

# Create image with modal_app mounted for imports
shape_generator_image = hunyuan_image.add_local_python_source("modal_app")

app = modal.App("hunyuan3d-shape-generator")


@app.cls(
    image=shape_generator_image,
    gpu="A10G",
    timeout=600,
    scaledown_window=300,
    enable_memory_snapshot=True,
)
class ShapeGenerator:
    """Modal-deployed shape generation service.

    Uses memory snapshots for fast cold starts:
    - snap=True: Loads models to CPU during image build (~25s)
    - snap=False: Moves models to GPU after restore (~5s)
    - Total cold start after snapshot: ~10-15s (vs ~45s without)
    """

    @modal.enter(snap=True)
    def load_models_to_cpu(self):
        """Load models to CPU memory (runs during image build for snapshotting)."""
        import os
        import sys

        # Set up Python paths for hy3dshape imports
        sys.path.insert(0, HY3DSHAPE_PACKAGE_PATH)
        os.chdir(APP_DIR)

        # Import pipeline and rembg
        from hy3dshape.rembg import BackgroundRemover

        from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline

        # Load pipeline to CPU (will be moved to GPU in snap=False)
        # Use HF repo ID so from_pretrained finds models in cache
        self._pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            HUNYUAN3D_REPO_ID,
            subfolder="hunyuan3d-dit-v2-1",
            use_safetensors=False,
            device="cpu",
        )

        # Load background remover
        self._rembg = BackgroundRemover()

        # Initialize service with dependencies via constructor injection
        self._service = ShapeGeneratorService(
            pipeline=self._pipeline,
            rembg=self._rembg,
        )

    @modal.enter(snap=False)
    def move_models_to_gpu(self):
        """Move models to GPU after restore from snapshot."""
        self._pipeline.to("cuda")

    @modal.method()
    def generate(
        self,
        image_bytes: bytes,
        seed: int | None = None,
        steps: int = 30,
        guidance_scale: float = 5.0,
        octree_resolution: int = 256,
    ) -> bytes:
        """Generate a 3D mesh from an input image.

        Args:
            image_bytes: PNG or JPEG image as bytes
            seed: Optional seed for reproducibility
            steps: Number of inference steps (default: 30, higher = better quality)
            guidance_scale: Classifier-free guidance scale (default: 5.0)
            octree_resolution: Mesh octree resolution (default: 256, higher = more detail)

        Returns:
            GLB mesh file as bytes
        """
        return self._service.generate(
            image_bytes,
            seed=seed,
            steps=steps,
            guidance_scale=guidance_scale,
            octree_resolution=octree_resolution,
        )
