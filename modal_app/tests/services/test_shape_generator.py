"""Unit tests for ShapeGenerator service.

Tests the shape generation service logic using mocks for the underlying pipeline.
These tests run without GPU and verify the orchestration logic:
- Image bytes loading and validation
- Background removal is applied
- Pipeline is called correctly
- Mesh export to GLB bytes
- Error handling

Following TDD: tests are written first, then implementation.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestShapeGeneratorInterface:
    """Test the ShapeGenerator interface contract.

    These tests verify the public interface using mocks for GPU components.
    Integration tests with real GPU are in tests/integration/.
    """

    def test_generate_accepts_image_bytes_and_returns_glb_bytes(
        self, sample_image_256, mocked_shape_service
    ):
        """The core contract: bytes in, bytes out."""
        service, _mocks = mocked_shape_service

        result = service.generate(sample_image_256)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_generate_with_seed_parameter(self, sample_image_256, mocked_shape_service):
        """Seed parameter should be accepted for reproducibility."""
        service, _mocks = mocked_shape_service

        result = service.generate(sample_image_256, seed=42)

        assert isinstance(result, bytes)

    def test_generate_raises_value_error_for_empty_bytes(self):
        """Empty image bytes should raise ValueError."""
        from modal_app.services.shape_generator import ShapeGeneratorService

        service = ShapeGeneratorService()

        with pytest.raises(ValueError, match="empty"):
            service.generate(b"")

    def test_generate_raises_value_error_for_invalid_image(self):
        """Invalid image data should raise ValueError."""
        from modal_app.services.shape_generator import ShapeGeneratorService

        service = ShapeGeneratorService()

        with pytest.raises(ValueError, match="invalid"):
            service.generate(b"not-a-valid-image")


class TestShapeGeneratorOrchestration:
    """Test the internal orchestration logic with mocks."""

    def test_image_bytes_converted_to_pil_image(
        self, sample_image_256, mocked_shape_service
    ):
        """Image bytes should be loaded into PIL Image."""
        service, _mocks = mocked_shape_service

        with patch.object(service, "_load_image") as mock_load:
            mock_load.return_value = MagicMock()

            service.generate(sample_image_256)

            mock_load.assert_called_once_with(sample_image_256)

    def test_background_removal_is_applied(
        self, sample_image_256, mocked_shape_service
    ):
        """Background removal should be called on the loaded image."""
        service, mocks = mocked_shape_service
        mock_image = MagicMock()

        with patch.object(service, "_load_image", return_value=mock_image):
            service.generate(sample_image_256)

            mocks["rembg"].assert_called_once_with(mock_image)

    def test_pipeline_receives_processed_image(
        self, sample_image_256, mocked_shape_service
    ):
        """Pipeline should receive the background-removed image."""
        service, mocks = mocked_shape_service
        mock_processed = MagicMock()
        mocks["rembg"].return_value = mock_processed

        service.generate(sample_image_256, seed=42)

        mocks["pipeline"].assert_called_once()
        call_args = mocks["pipeline"].call_args
        assert call_args[1]["image"] == mock_processed
        assert call_args[1]["seed"] == 42

    def test_mesh_exported_to_glb_bytes(self, sample_image_256, mocked_shape_service):
        """Pipeline output mesh should be exported to GLB bytes."""
        service, mocks = mocked_shape_service
        mock_mesh = MagicMock()
        mocks["pipeline"].return_value = mock_mesh

        result = service.generate(sample_image_256)

        mocks["export"].assert_called_once_with(mock_mesh)
        assert result == b"glb-bytes"


class TestShapeGeneratorValidation:
    """Test input validation."""

    def test_validates_image_can_be_loaded(self):
        """Should validate image bytes can be decoded."""
        from modal_app.services.shape_generator import ShapeGeneratorService

        service = ShapeGeneratorService()

        with pytest.raises(ValueError, match="invalid"):
            service.generate(b"\x00\x00\x00\x00")

    def test_handles_rgb_images(self, sample_image_256, mocked_shape_service):
        """RGB images should be handled (background removal converts to RGBA)."""
        service, _mocks = mocked_shape_service

        result = service.generate(sample_image_256)

        assert result == b"glb-bytes"


class TestShapeGeneratorParameters:
    """Test generation parameter handling."""

    def test_generate_accepts_steps_parameter(
        self, sample_image_256, mocked_shape_service
    ):
        """Steps parameter should be accepted and passed to pipeline."""
        service, mocks = mocked_shape_service

        service.generate(sample_image_256, steps=50)

        call_kwargs = mocks["pipeline"].call_args[1]
        assert call_kwargs["steps"] == 50

    def test_generate_uses_default_steps_when_not_provided(
        self, sample_image_256, mocked_shape_service
    ):
        """Default steps value should be 30."""
        service, mocks = mocked_shape_service

        service.generate(sample_image_256)

        call_kwargs = mocks["pipeline"].call_args[1]
        assert call_kwargs["steps"] == 30

    def test_generate_accepts_guidance_scale_parameter(
        self, sample_image_256, mocked_shape_service
    ):
        """Guidance scale parameter should be accepted and passed to pipeline."""
        service, mocks = mocked_shape_service

        service.generate(sample_image_256, guidance_scale=7.5)

        call_kwargs = mocks["pipeline"].call_args[1]
        assert call_kwargs["guidance_scale"] == 7.5

    def test_generate_uses_default_guidance_scale_when_not_provided(
        self, sample_image_256, mocked_shape_service
    ):
        """Default guidance_scale value should be 5.0."""
        service, mocks = mocked_shape_service

        service.generate(sample_image_256)

        call_kwargs = mocks["pipeline"].call_args[1]
        assert call_kwargs["guidance_scale"] == 5.0

    def test_generate_accepts_octree_resolution_parameter(
        self, sample_image_256, mocked_shape_service
    ):
        """Octree resolution parameter should be accepted and passed to pipeline."""
        service, mocks = mocked_shape_service

        service.generate(sample_image_256, octree_resolution=384)

        call_kwargs = mocks["pipeline"].call_args[1]
        assert call_kwargs["octree_resolution"] == 384

    def test_generate_uses_default_octree_resolution_when_not_provided(
        self, sample_image_256, mocked_shape_service
    ):
        """Default octree_resolution value should be 256."""
        service, mocks = mocked_shape_service

        service.generate(sample_image_256)

        call_kwargs = mocks["pipeline"].call_args[1]
        assert call_kwargs["octree_resolution"] == 256

    def test_generate_passes_all_parameters_together(
        self, sample_image_256, mocked_shape_service
    ):
        """All parameters should work together."""
        service, mocks = mocked_shape_service

        service.generate(
            sample_image_256,
            seed=42,
            steps=25,
            guidance_scale=3.0,
            octree_resolution=512,
        )

        call_kwargs = mocks["pipeline"].call_args[1]
        assert call_kwargs["seed"] == 42
        assert call_kwargs["steps"] == 25
        assert call_kwargs["guidance_scale"] == 3.0
        assert call_kwargs["octree_resolution"] == 512
