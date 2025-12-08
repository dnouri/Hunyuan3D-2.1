"""Unit tests for TextureGenerator service.

Tests the texture generation service logic using mocks for the underlying pipeline.
These tests run without GPU and verify the orchestration logic:
- Mesh bytes loading and validation
- Image bytes loading and validation
- Texture pipeline is called correctly
- Textured mesh export to GLB bytes
- Error handling

Following TDD: tests are written first, then implementation.
"""

from unittest.mock import patch

import pytest


class TestTextureGeneratorInterface:
    """Test the TextureGenerator interface contract.

    These tests verify the public interface using mocks for GPU components.
    Integration tests with real GPU are in tests/integration/.
    """

    def test_generate_accepts_mesh_and_image_bytes_returns_glb_bytes(
        self, sample_glb_bytes, sample_image_256, mocked_texture_service
    ):
        """The core contract: mesh bytes + image bytes in, textured mesh bytes out."""
        service, _mocks = mocked_texture_service

        result = service.generate(sample_glb_bytes, sample_image_256)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_generate_with_seed_parameter(
        self, sample_glb_bytes, sample_image_256, mocked_texture_service
    ):
        """Seed parameter should be accepted for reproducibility."""
        service, _mocks = mocked_texture_service

        result = service.generate(sample_glb_bytes, sample_image_256, seed=42)

        assert isinstance(result, bytes)

    def test_generate_raises_value_error_for_empty_mesh_bytes(self):
        """Empty mesh bytes should raise ValueError."""
        from modal_app.services.texture_generator import TextureGeneratorService

        service = TextureGeneratorService()

        with pytest.raises(ValueError, match="mesh_bytes.*empty"):
            service.generate(b"", b"image-data")

    def test_generate_raises_value_error_for_empty_image_bytes(self, sample_glb_bytes):
        """Empty image bytes should raise ValueError."""
        from modal_app.services.texture_generator import TextureGeneratorService

        service = TextureGeneratorService()

        with pytest.raises(ValueError, match="image_bytes.*empty"):
            service.generate(sample_glb_bytes, b"")

    def test_generate_raises_value_error_for_invalid_mesh(self, sample_image_256):
        """Invalid mesh data should raise ValueError."""
        from modal_app.services.texture_generator import TextureGeneratorService

        service = TextureGeneratorService()

        with pytest.raises(ValueError, match="invalid mesh"):
            service.generate(b"not-a-valid-mesh", sample_image_256)

    def test_generate_raises_value_error_for_invalid_image(self, sample_glb_bytes):
        """Invalid image data should raise ValueError."""
        from modal_app.services.texture_generator import TextureGeneratorService

        service = TextureGeneratorService()

        with pytest.raises(ValueError, match="invalid image"):
            service.generate(sample_glb_bytes, b"not-a-valid-image")


class TestTextureGeneratorOrchestration:
    """Test the internal orchestration logic with mocks."""

    def test_mesh_bytes_written_to_temp_file(
        self, sample_glb_bytes, sample_image_256, mocked_texture_service
    ):
        """Mesh bytes should be written to a temp file for the pipeline."""
        service, mocks = mocked_texture_service

        with patch.object(service, "_write_mesh_to_temp") as mock_write:
            mock_write.return_value = "/tmp/mesh.glb"

            service.generate(sample_glb_bytes, sample_image_256)

            mock_write.assert_called_once()
            call_args = mock_write.call_args
            assert call_args[0][0] == sample_glb_bytes

    def test_image_bytes_written_to_temp_file(
        self, sample_glb_bytes, sample_image_256, mocked_texture_service
    ):
        """Image bytes should be written to a temp file for the pipeline."""
        service, mocks = mocked_texture_service

        with patch.object(service, "_write_image_to_temp") as mock_write:
            mock_write.return_value = "/tmp/image.png"

            service.generate(sample_glb_bytes, sample_image_256)

            mock_write.assert_called_once()
            call_args = mock_write.call_args
            assert call_args[0][0] == sample_image_256

    def test_pipeline_receives_file_paths(
        self, sample_glb_bytes, sample_image_256, mocked_texture_service
    ):
        """Pipeline should receive the temp file paths."""
        service, mocks = mocked_texture_service
        mock_mesh_path = "/tmp/test/mesh.glb"
        mock_image_path = "/tmp/test/image.png"

        with patch.object(service, "_write_mesh_to_temp", return_value=mock_mesh_path):
            with patch.object(
                service, "_write_image_to_temp", return_value=mock_image_path
            ):
                service.generate(sample_glb_bytes, sample_image_256, seed=42)

                mocks["pipeline"].assert_called_once()
                call_kwargs = mocks["pipeline"].call_args[1]
                assert call_kwargs["mesh_path"] == mock_mesh_path
                assert call_kwargs["image_path"] == mock_image_path

    def test_output_glb_read_and_returned(
        self, sample_glb_bytes, sample_image_256, mocked_texture_service
    ):
        """Output GLB file should be read and returned as bytes."""
        service, mocks = mocked_texture_service
        expected_output = b"textured-glb-output"
        mocks["read_output"].return_value = expected_output

        result = service.generate(sample_glb_bytes, sample_image_256)

        assert result == expected_output


class TestTextureGeneratorValidation:
    """Test input validation."""

    def test_validates_mesh_has_glb_header(self, sample_image_256):
        """Should validate mesh bytes have GLB magic header."""
        from modal_app.services.texture_generator import TextureGeneratorService

        service = TextureGeneratorService()

        # Invalid header (not 'glTF')
        with pytest.raises(ValueError, match="invalid mesh"):
            service.generate(b"\x00\x00\x00\x00", sample_image_256)

    def test_validates_image_can_be_decoded(self, sample_glb_bytes):
        """Should validate image bytes can be decoded."""
        from modal_app.services.texture_generator import TextureGeneratorService

        service = TextureGeneratorService()

        with pytest.raises(ValueError, match="invalid image"):
            service.generate(sample_glb_bytes, b"\x00\x00\x00\x00")

    def test_accepts_valid_glb_and_png(
        self, sample_glb_bytes, sample_image_256, mocked_texture_service
    ):
        """Valid GLB and PNG should be accepted."""
        service, _mocks = mocked_texture_service

        result = service.generate(sample_glb_bytes, sample_image_256)

        assert result == b"textured-glb-output"
