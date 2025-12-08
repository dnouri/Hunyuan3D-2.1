"""FastAPI application factory for Hunyuan3D API.

Creates the ASGI application with dependency injection for services.
This allows mocking services in tests while using real Modal services in production.

Usage:
    # Production (Modal deployment)
    from modal_app.api.app import create_app
    app = create_app()  # Uses real services

    # Testing
    app = create_app(
        shape_generator=mock_shape_generator,
        texture_generator=mock_texture_generator,
        s3_storage=mock_s3_storage,
    )
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Protocol

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import StreamingResponse

from modal_app.api.auth import verify_api_key
from modal_app.api.sse import (
    sse_completed,
    sse_error,
    sse_progress,
    sse_started,
)


class GenerateMethod(Protocol):
    """Protocol for service generate methods (.remote() pattern)."""

    def remote(self, *args: Any, **kwargs: Any) -> bytes: ...


class ShapeGeneratorProtocol(Protocol):
    """Protocol for shape generator services."""

    @property
    def generate(self) -> GenerateMethod: ...


class TextureGeneratorProtocol(Protocol):
    """Protocol for texture generator services."""

    @property
    def generate(self) -> GenerateMethod: ...


class StorageProtocol(Protocol):
    """Protocol for storage services."""

    def upload_artifact(self, job_id: str, data: bytes, content_type: str) -> str: ...


def create_app(
    *,
    shape_generator: ShapeGeneratorProtocol | None = None,
    texture_generator: TextureGeneratorProtocol | None = None,
    s3_storage: StorageProtocol | None = None,
) -> FastAPI:
    """Create the FastAPI application with optional dependency injection.

    Args:
        shape_generator: Shape generation service (uses real Modal service if None)
        texture_generator: Texture generation service (uses real Modal service if None)
        s3_storage: S3 storage module (uses real S3 if None)

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="Hunyuan3D-2.1 API",
        description="Generate 3D meshes from images using Hunyuan3D-2.1",
        version="1.0.0",
    )

    # Store dependencies for routes to access
    app.state.shape_generator = shape_generator
    app.state.texture_generator = texture_generator
    app.state.s3_storage = s3_storage

    @app.get("/health")
    async def health():
        """Health check endpoint (no authentication required)."""
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @app.post("/generate/stream")
    async def generate_stream(
        api_key: str = verify_api_key,
        image: UploadFile = File(...),
        generate_texture: bool = Form(True),
        seed: int | None = Form(None),
        steps: int = Form(30, ge=10, le=100),
        guidance_scale: float = Form(5.0, ge=1.0, le=15.0),
        octree_resolution: int = Form(256),
    ):
        """Generate a 3D mesh from an image with SSE progress streaming.

        Args:
            api_key: Valid API key (from X-API-Key header)
            image: Input image file (PNG or JPEG)
            generate_texture: Whether to generate PBR textures (default: true)
            seed: Optional seed for reproducibility
            steps: Number of inference steps (10-100, default: 30)
            guidance_scale: Classifier-free guidance scale (1.0-15.0, default: 5.0)
            octree_resolution: Mesh octree resolution (default: 256)

        Returns:
            SSE stream with progress events and final download URL

        Note:
            Shape generation parameters (steps, guidance_scale, octree_resolution) are
            configurable per-request. Texture quality settings (views, resolution) are
            fixed at service deployment time due to pipeline architecture constraints.
        """
        # Read image bytes
        image_bytes = await image.read()

        # Generate unique job ID
        job_id = str(uuid.uuid4())

        async def event_stream():
            yield sse_started(job_id)

            try:
                # Shape generation
                yield sse_progress("shape", 10)

                shape_gen = app.state.shape_generator
                mesh_bytes = shape_gen.generate.remote(
                    image_bytes,
                    seed=seed,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    octree_resolution=octree_resolution,
                )

                yield sse_progress("shape_complete", 50)

                # Texture generation (optional)
                if generate_texture:
                    yield sse_progress("texture", 55)

                    texture_gen = app.state.texture_generator
                    result_bytes = texture_gen.generate.remote(
                        mesh_bytes, image_bytes, seed=seed
                    )

                    yield sse_progress("texture_complete", 95)
                else:
                    result_bytes = mesh_bytes

                # Upload to S3
                yield sse_progress("uploading", 98)

                storage = app.state.s3_storage
                download_url = storage.upload_artifact(
                    job_id, result_bytes, "model/gltf-binary"
                )

                yield sse_completed(job_id, download_url)

            except Exception as e:
                yield sse_error("generation", str(e), retriable=False)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app
