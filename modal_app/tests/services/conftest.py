"""Fixtures for service unit tests.

Provides common mocking patterns for GPU service tests.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mocked_shape_service():
    """Yield ShapeGeneratorService with GPU methods mocked.

    Returns:
        Tuple of (service, mocks_dict) where mocks_dict contains:
        - "rembg": mock for _remove_background
        - "pipeline": mock for _run_pipeline
        - "export": mock for _export_mesh

    Usage:
        def test_something(mocked_shape_service):
            service, mocks = mocked_shape_service
            result = service.generate(image_bytes)
            mocks["pipeline"].assert_called_once()
    """
    from modal_app.services.shape_generator import ShapeGeneratorService

    service = ShapeGeneratorService()

    with (
        patch.object(service, "_remove_background") as mock_rembg,
        patch.object(service, "_run_pipeline") as mock_pipeline,
        patch.object(service, "_export_mesh") as mock_export,
    ):
        # Set up default return values
        mock_rembg.return_value = MagicMock(name="processed_image")
        mock_pipeline.return_value = MagicMock(name="mesh")
        mock_export.return_value = b"glb-bytes"

        yield (
            service,
            {
                "rembg": mock_rembg,
                "pipeline": mock_pipeline,
                "export": mock_export,
            },
        )


@pytest.fixture
def mocked_texture_service():
    """Yield TextureGeneratorService with GPU methods mocked.

    Returns:
        Tuple of (service, mocks_dict) where mocks_dict contains:
        - "pipeline": mock for _run_pipeline
        - "read_output": mock for _read_output_glb

    Usage:
        def test_something(mocked_texture_service):
            service, mocks = mocked_texture_service
            result = service.generate(mesh_bytes, image_bytes)
            mocks["pipeline"].assert_called_once()
    """
    from modal_app.services.texture_generator import TextureGeneratorService

    service = TextureGeneratorService()

    with (
        patch.object(service, "_write_mesh_to_temp") as mock_write_mesh,
        patch.object(service, "_write_image_to_temp") as mock_write_image,
        patch.object(service, "_run_pipeline") as mock_pipeline,
        patch.object(service, "_read_output_glb") as mock_read_output,
    ):
        # Set up default return values
        mock_write_mesh.return_value = "/tmp/mock/mesh.glb"
        mock_write_image.return_value = "/tmp/mock/image.png"
        mock_pipeline.return_value = "/tmp/mock/textured.glb"
        mock_read_output.return_value = b"textured-glb-output"

        yield (
            service,
            {
                "write_mesh": mock_write_mesh,
                "write_image": mock_write_image,
                "pipeline": mock_pipeline,
                "read_output": mock_read_output,
            },
        )
