"""End-to-end tests for the full Hunyuan3D pipeline.

These tests verify the complete generation flow using mocks for fast,
repeatable local testing. They cover all major scenarios:
- Shape only generation
- Shape + texture generation
- Various image sizes
- Invalid inputs
- Authentication

Run with:
    pytest modal_app/tests/e2e/test_full_pipeline.py -v
"""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from modal_app.tests.conftest import parse_sse_events

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def valid_api_key():
    return "sk_test_e2e_testing_key_12345678901"


@pytest.fixture
def sample_image_256():
    return (FIXTURES_DIR / "test_image_256x256.png").read_bytes()


@pytest.fixture
def sample_image_1x1():
    return (FIXTURES_DIR / "test_image_1x1.png").read_bytes()


@pytest.fixture
def mock_shape_generator():
    from modal_app.tests.mocks.modal_services import MockShapeGenerator

    return MockShapeGenerator(delay=0)


@pytest.fixture
def mock_texture_generator():
    from modal_app.tests.mocks.modal_services import MockTextureGenerator

    return MockTextureGenerator(delay=0)


@pytest.fixture
def mock_s3_storage():
    from modal_app.tests.mocks.modal_services import MockS3Storage

    return MockS3Storage()


@pytest.fixture
def app(
    mock_shape_generator,
    mock_texture_generator,
    mock_s3_storage,
    valid_api_key,
    monkeypatch,
):
    monkeypatch.setenv("VALID_API_KEYS", valid_api_key)
    from modal_app.api.app import create_app

    return create_app(
        shape_generator=mock_shape_generator,
        texture_generator=mock_texture_generator,
        s3_storage=mock_s3_storage,
    )


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def auth_headers(valid_api_key):
    return {"X-API-Key": valid_api_key}


class TestShapeOnlyGeneration:
    """Tests for shape-only generation (generate_texture=false)."""

    def test_shape_only_simple_image(self, client, auth_headers, sample_image_256):
        """Shape generation with standard 256x256 image."""
        response = client.post(
            "/generate/stream",
            headers=auth_headers,
            files={"image": ("test.png", io.BytesIO(sample_image_256), "image/png")},
            data={"generate_texture": "false"},
        )

        assert response.status_code == 200
        events = parse_sse_events(response.text)

        # Verify event sequence
        event_types = [e["event"] for e in events]
        assert event_types[0] == "started"
        assert "progress" in event_types
        assert event_types[-1] == "completed"

        # Verify no texture stages
        stages = [e["data"].get("stage") for e in events if e["event"] == "progress"]
        assert not any("texture" in (s or "") for s in stages)

    def test_shape_only_small_image(self, client, auth_headers, sample_image_1x1):
        """Shape generation with minimal 1x1 image."""
        response = client.post(
            "/generate/stream",
            headers=auth_headers,
            files={"image": ("tiny.png", io.BytesIO(sample_image_1x1), "image/png")},
            data={"generate_texture": "false"},
        )

        assert response.status_code == 200
        events = parse_sse_events(response.text)
        assert any(e["event"] == "completed" for e in events)

    def test_shape_only_returns_download_url(
        self, client, auth_headers, sample_image_256
    ):
        """Completed event should contain download URL."""
        response = client.post(
            "/generate/stream",
            headers=auth_headers,
            files={"image": ("test.png", io.BytesIO(sample_image_256), "image/png")},
            data={"generate_texture": "false"},
        )

        events = parse_sse_events(response.text)
        completed = next(e for e in events if e["event"] == "completed")

        assert "download_url" in completed["data"]
        assert "job_id" in completed["data"]
        assert ".glb" in completed["data"]["download_url"]


class TestFullGeneration:
    """Tests for full generation (shape + texture)."""

    def test_full_generation_simple_image(self, client, auth_headers, sample_image_256):
        """Full generation with standard 256x256 image."""
        response = client.post(
            "/generate/stream",
            headers=auth_headers,
            files={"image": ("test.png", io.BytesIO(sample_image_256), "image/png")},
            data={"generate_texture": "true"},
        )

        assert response.status_code == 200
        events = parse_sse_events(response.text)

        # Verify event sequence
        event_types = [e["event"] for e in events]
        assert event_types[0] == "started"
        assert event_types[-1] == "completed"

        # Verify texture stages present
        stages = [e["data"].get("stage") for e in events if e["event"] == "progress"]
        assert any("shape" in (s or "") for s in stages)
        assert any("texture" in (s or "") for s in stages)

    def test_full_generation_default_is_true(
        self, client, auth_headers, sample_image_256
    ):
        """Without generate_texture param, should default to true."""
        response = client.post(
            "/generate/stream",
            headers=auth_headers,
            files={"image": ("test.png", io.BytesIO(sample_image_256), "image/png")},
        )

        events = parse_sse_events(response.text)
        stages = [e["data"].get("stage") for e in events if e["event"] == "progress"]

        # Should have texture stages by default
        assert any("texture" in (s or "") for s in stages)

    def test_full_generation_with_seed(
        self, client, auth_headers, sample_image_256, mock_shape_generator
    ):
        """Generation with seed parameter should pass through."""
        response = client.post(
            "/generate/stream",
            headers=auth_headers,
            files={"image": ("test.png", io.BytesIO(sample_image_256), "image/png")},
            data={"generate_texture": "true", "seed": "42"},
        )

        assert response.status_code == 200
        events = parse_sse_events(response.text)
        assert any(e["event"] == "completed" for e in events)


class TestProgressEvents:
    """Tests for SSE progress event sequence."""

    def test_progress_percentage_increases(
        self, client, auth_headers, sample_image_256
    ):
        """Progress percentages should increase monotonically."""
        response = client.post(
            "/generate/stream",
            headers=auth_headers,
            files={"image": ("test.png", io.BytesIO(sample_image_256), "image/png")},
        )

        events = parse_sse_events(response.text)
        percentages = [e["data"]["percent"] for e in events if e["event"] == "progress"]

        # Should be monotonically increasing
        for i in range(1, len(percentages)):
            assert percentages[i] >= percentages[i - 1]

    def test_progress_stages_are_descriptive(
        self, client, auth_headers, sample_image_256
    ):
        """Progress stages should have meaningful names."""
        response = client.post(
            "/generate/stream",
            headers=auth_headers,
            files={"image": ("test.png", io.BytesIO(sample_image_256), "image/png")},
        )

        events = parse_sse_events(response.text)
        stages = [e["data"]["stage"] for e in events if e["event"] == "progress"]

        # All stages should be non-empty strings
        for stage in stages:
            assert isinstance(stage, str)
            assert len(stage) > 0


class TestJobId:
    """Tests for job ID generation and consistency."""

    def test_job_id_is_uuid_format(self, client, auth_headers, sample_image_256):
        """Job ID should be a valid UUID."""
        import uuid

        response = client.post(
            "/generate/stream",
            headers=auth_headers,
            files={"image": ("test.png", io.BytesIO(sample_image_256), "image/png")},
        )

        events = parse_sse_events(response.text)
        started = next(e for e in events if e["event"] == "started")
        job_id = started["data"]["job_id"]

        # Should parse as UUID without error
        uuid.UUID(job_id)

    def test_job_id_consistent_across_events(
        self, client, auth_headers, sample_image_256
    ):
        """Job ID in started and completed should match."""
        response = client.post(
            "/generate/stream",
            headers=auth_headers,
            files={"image": ("test.png", io.BytesIO(sample_image_256), "image/png")},
        )

        events = parse_sse_events(response.text)
        started = next(e for e in events if e["event"] == "started")
        completed = next(e for e in events if e["event"] == "completed")

        assert started["data"]["job_id"] == completed["data"]["job_id"]


class TestAuthentication:
    """Tests for API key authentication."""

    def test_missing_api_key_returns_401(self, client, sample_image_256):
        """Request without API key should be rejected."""
        response = client.post(
            "/generate/stream",
            files={"image": ("test.png", io.BytesIO(sample_image_256), "image/png")},
        )

        assert response.status_code == 401
        assert "Missing" in response.json()["detail"]

    def test_invalid_api_key_returns_401(self, client, sample_image_256):
        """Request with wrong API key should be rejected."""
        response = client.post(
            "/generate/stream",
            headers={"X-API-Key": "wrong-key"},
            files={"image": ("test.png", io.BytesIO(sample_image_256), "image/png")},
        )

        assert response.status_code == 401
        assert "Invalid" in response.json()["detail"]


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status(self, client):
        """Health response should contain status."""
        response = client.get("/health")
        assert response.json()["status"] == "healthy"

    def test_health_no_auth_required(self, client):
        """Health endpoint should not require authentication."""
        response = client.get("/health")
        assert response.status_code == 200
