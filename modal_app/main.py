"""Modal entry point for Hunyuan3D-2.1 API.

This module defines the Modal app that serves the FastAPI API.
It connects the API to the real GPU services (ShapeGenerator, TextureGenerator)
and S3 storage.

Deployment:
    modal deploy modal_app/main.py

Development:
    modal serve modal_app/main.py
"""

from __future__ import annotations

import modal

from modal_app.image import hunyuan_image

# Create the main app
app = modal.App("hunyuan3d-api")

# Create image with modal_app mounted
api_image = hunyuan_image.add_local_python_source("modal_app")


class ProductionShapeGenerator:
    """Wrapper that provides .generate.remote() interface for production.

    In production, we look up the deployed ShapeGenerator service.
    """

    def __init__(self):
        from modal_app.services.shape_generator import ShapeGenerator

        self._cls = ShapeGenerator

    @property
    def generate(self):
        return self._cls().generate


class ProductionTextureGenerator:
    """Wrapper that provides .generate.remote() interface for production.

    In production, we look up the deployed TextureGenerator service.
    """

    def __init__(self):
        from modal_app.services.texture_generator import TextureGenerator

        self._cls = TextureGenerator

    @property
    def generate(self):
        return self._cls().generate


class ProductionS3Storage:
    """Wrapper for S3 storage functions."""

    def upload_artifact(self, job_id: str, data: bytes, content_type: str) -> str:
        from modal_app.storage.s3 import upload_artifact

        return upload_artifact(job_id, data, content_type)

    def generate_presigned_url(self, job_id: str, expires_in: int = 3600) -> str:
        from modal_app.storage.s3 import generate_presigned_url

        return generate_presigned_url(job_id, expires_in)


@app.function(
    image=api_image,
    secrets=[
        modal.Secret.from_name("aws-credentials"),
        modal.Secret.from_name("api-keys"),
    ],
    timeout=600,
    allow_concurrent_inputs=100,
)
@modal.asgi_app()
def api():
    """Create and return the FastAPI application.

    The GPU services are accessed via their deployed Modal endpoints.
    S3 storage uses credentials from Modal secrets.
    """
    from modal_app.api.app import create_app

    return create_app(
        shape_generator=ProductionShapeGenerator(),
        texture_generator=ProductionTextureGenerator(),
        s3_storage=ProductionS3Storage(),
    )


@app.local_entrypoint()
def main():
    """Local entrypoint for testing/debugging."""
    print("=" * 60)
    print("Hunyuan3D-2.1 API")
    print("=" * 60)
    print()
    print("To run the API locally with hot-reload:")
    print("  modal serve modal_app/main.py")
    print()
    print("To deploy to production:")
    print("  modal deploy modal_app/main.py")
    print()
    print("API Endpoints:")
    print("  GET  /health           - Health check (no auth)")
    print("  POST /generate/stream  - Generate 3D mesh (SSE)")
    print()
    print("Required Modal Secrets:")
    print("  - aws-credentials: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,")
    print("                     AWS_REGION, S3_BUCKET_NAME")
    print("  - api-keys: VALID_API_KEYS (comma-separated)")
    print("=" * 60)
