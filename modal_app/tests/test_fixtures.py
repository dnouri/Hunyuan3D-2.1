"""Tests to verify test fixtures and mocks work correctly.

These tests ensure our test infrastructure is properly set up before
we start writing production code.
"""

import pytest


class TestImageFixtures:
    """Verify image fixtures are valid PNGs."""

    def test_sample_image_bytes_is_valid_png(self, sample_image_bytes: bytes) -> None:
        """Sample image should have PNG signature."""
        assert sample_image_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    def test_sample_image_256_is_valid_png(self, sample_image_256: bytes) -> None:
        """256x256 image should have PNG signature."""
        assert sample_image_256[:8] == b"\x89PNG\r\n\x1a\n"

    def test_sample_image_256_is_larger(
        self, sample_image_bytes: bytes, sample_image_256: bytes
    ) -> None:
        """256x256 image should be larger than 1x1."""
        assert len(sample_image_256) > len(sample_image_bytes)


class TestGLBFixtures:
    """Verify GLB fixtures are valid."""

    def test_sample_glb_has_correct_magic(self, sample_glb_bytes: bytes) -> None:
        """GLB should start with 'glTF' magic bytes."""
        assert sample_glb_bytes[:4] == b"glTF"

    def test_sample_glb_has_version_2(self, sample_glb_bytes: bytes) -> None:
        """GLB should be version 2."""
        import struct

        version = struct.unpack("<I", sample_glb_bytes[4:8])[0]
        assert version == 2


class TestMockShapeGenerator:
    """Verify MockShapeGenerator behaves correctly."""

    def test_generate_remote_returns_bytes(
        self, mock_shape_generator, sample_image_bytes: bytes
    ) -> None:
        """generate.remote() should return bytes (Modal pattern)."""
        result = mock_shape_generator.generate.remote(sample_image_bytes)
        assert isinstance(result, bytes)

    def test_generate_direct_returns_bytes(
        self, mock_shape_generator, sample_image_bytes: bytes
    ) -> None:
        """generate() direct call should also work."""
        result = mock_shape_generator.generate(sample_image_bytes)
        assert isinstance(result, bytes)

    def test_generate_returns_valid_glb(
        self, mock_shape_generator, sample_image_bytes: bytes
    ) -> None:
        """generate() should return valid GLB."""
        result = mock_shape_generator.generate.remote(sample_image_bytes)
        assert result[:4] == b"glTF"

    def test_generate_tracks_call_count(
        self, mock_shape_generator, sample_image_bytes: bytes
    ) -> None:
        """Call count should increment with each call."""
        assert mock_shape_generator.call_count == 0
        mock_shape_generator.generate.remote(sample_image_bytes)
        assert mock_shape_generator.call_count == 1
        mock_shape_generator.generate.remote(sample_image_bytes)
        assert mock_shape_generator.call_count == 2

    def test_generate_raises_on_empty_input(self, mock_shape_generator) -> None:
        """generate() should raise ValueError for empty input."""
        with pytest.raises(ValueError, match="cannot be empty"):
            mock_shape_generator.generate.remote(b"")

    def test_generate_raises_on_invalid_input(self, mock_shape_generator) -> None:
        """generate() should raise ValueError for too-small input."""
        with pytest.raises(ValueError, match="too small"):
            mock_shape_generator.generate.remote(b"tiny")

    def test_fail_on_call_raises_error(self, sample_image_bytes: bytes) -> None:
        """fail_on_call should trigger error on specified call."""
        from modal_app.tests.mocks.modal_services import MockShapeGenerator

        mock = MockShapeGenerator(fail_on_call=2, error_message="Test error")
        mock.generate.remote(sample_image_bytes)
        with pytest.raises(RuntimeError, match="Test error"):
            mock.generate.remote(sample_image_bytes)

    def test_spawn_returns_handle(
        self, mock_shape_generator, sample_image_bytes: bytes
    ) -> None:
        """generate.spawn() should return handle for async pattern."""
        handle = mock_shape_generator.generate.spawn(sample_image_bytes)
        result = handle.get()
        assert isinstance(result, bytes)
        assert result[:4] == b"glTF"


class TestMockTextureGenerator:
    """Verify MockTextureGenerator behaves correctly."""

    def test_generate_remote_returns_bytes(
        self, mock_texture_generator, sample_glb_bytes: bytes, sample_image_bytes: bytes
    ) -> None:
        """generate.remote() should return bytes (Modal pattern)."""
        result = mock_texture_generator.generate.remote(
            sample_glb_bytes, sample_image_bytes
        )
        assert isinstance(result, bytes)

    def test_generate_raises_on_empty_mesh(
        self, mock_texture_generator, sample_image_bytes: bytes
    ) -> None:
        """generate() should raise ValueError for empty mesh."""
        with pytest.raises(ValueError, match="mesh_bytes cannot be empty"):
            mock_texture_generator.generate.remote(b"", sample_image_bytes)

    def test_generate_raises_on_empty_image(
        self, mock_texture_generator, sample_glb_bytes: bytes
    ) -> None:
        """generate() should raise ValueError for empty image."""
        with pytest.raises(ValueError, match="image_bytes cannot be empty"):
            mock_texture_generator.generate.remote(sample_glb_bytes, b"")


class TestMockS3Storage:
    """Verify MockS3Storage behaves correctly."""

    def test_upload_returns_url(
        self, mock_s3_storage, job_id: str, sample_glb_bytes: bytes
    ) -> None:
        """upload_artifact() should return a URL."""
        url = mock_s3_storage.upload_artifact(job_id, sample_glb_bytes)
        assert url.startswith("https://")
        assert job_id in url

    def test_uploaded_data_retrievable(
        self, mock_s3_storage, job_id: str, sample_glb_bytes: bytes
    ) -> None:
        """Uploaded data should be retrievable for verification."""
        mock_s3_storage.upload_artifact(job_id, sample_glb_bytes)
        retrieved = mock_s3_storage.get_upload(job_id)
        assert retrieved == sample_glb_bytes

    def test_presigned_url_contains_job_id(self, mock_s3_storage, job_id: str) -> None:
        """Presigned URL should contain job ID."""
        url = mock_s3_storage.generate_presigned_url(job_id)
        assert job_id in url
