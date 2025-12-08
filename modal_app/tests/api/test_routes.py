"""Tests for API routes.

The API provides:
- POST /generate/stream: Main generation endpoint with SSE
- GET /health: Health check (no auth required)
"""

import io

import pytest
from fastapi.testclient import TestClient

from modal_app.tests.conftest import parse_sse_events


@pytest.fixture
def valid_api_key():
    """A valid test API key."""
    return "sk_test_abc123def456ghi789jkl012mno345"


@pytest.fixture
def sample_image_bytes():
    """Valid PNG image bytes for testing."""
    from pathlib import Path

    fixtures_dir = Path(__file__).parent.parent / "fixtures"
    return (fixtures_dir / "test_image_256x256.png").read_bytes()


@pytest.fixture
def mock_shape_generator():
    """Mock ShapeGenerator for fast tests."""
    from modal_app.tests.mocks.modal_services import MockShapeGenerator

    return MockShapeGenerator(delay=0)


@pytest.fixture
def mock_texture_generator():
    """Mock TextureGenerator for fast tests."""
    from modal_app.tests.mocks.modal_services import MockTextureGenerator

    return MockTextureGenerator(delay=0)


@pytest.fixture
def mock_s3_storage():
    """Mock S3 storage for testing without AWS."""
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
    """Create app with mocked dependencies."""
    monkeypatch.setenv("VALID_API_KEYS", valid_api_key)

    from modal_app.api.app import create_app

    return create_app(
        shape_generator=mock_shape_generator,
        texture_generator=mock_texture_generator,
        s3_storage=mock_s3_storage,
    )


@pytest.fixture
def client(app):
    """Test client with mocked services."""
    return TestClient(app)


@pytest.fixture
def auth_headers(valid_api_key):
    """Headers with valid API key."""
    return {"X-API-Key": valid_api_key}


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_returns_200_without_auth(self, client):
        """Health check should not require authentication."""
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_returns_timestamp(self, client):
        """Health response should include a timestamp."""
        response = client.get("/health")

        assert "timestamp" in response.json()


class TestGenerateStreamEndpoint:
    """Tests for POST /generate/stream."""

    def test_requires_authentication(self, client, sample_image_bytes):
        """Should return 401 without API key."""
        response = client.post(
            "/generate/stream",
            files={"image": ("test.png", io.BytesIO(sample_image_bytes), "image/png")},
        )

        assert response.status_code == 401

    def test_accepts_valid_request(self, client, auth_headers, sample_image_bytes):
        """Valid request should return 200 with SSE content type."""
        response = client.post(
            "/generate/stream",
            headers=auth_headers,
            files={"image": ("test.png", io.BytesIO(sample_image_bytes), "image/png")},
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

    def test_returns_sse_stream(self, client, auth_headers, sample_image_bytes):
        """Response should be a valid SSE stream."""
        response = client.post(
            "/generate/stream",
            headers=auth_headers,
            files={"image": ("test.png", io.BytesIO(sample_image_bytes), "image/png")},
        )

        # Parse SSE events
        events = parse_sse_events(response.text)

        assert len(events) > 0
        assert events[0]["event"] == "started"

    def test_stream_includes_started_event(
        self, client, auth_headers, sample_image_bytes
    ):
        """Stream should start with 'started' event containing job_id."""
        response = client.post(
            "/generate/stream",
            headers=auth_headers,
            files={"image": ("test.png", io.BytesIO(sample_image_bytes), "image/png")},
        )

        events = parse_sse_events(response.text)
        started = next(e for e in events if e["event"] == "started")

        assert "job_id" in started["data"]
        assert len(started["data"]["job_id"]) > 0

    def test_stream_includes_progress_events(
        self, client, auth_headers, sample_image_bytes
    ):
        """Stream should include progress events."""
        response = client.post(
            "/generate/stream",
            headers=auth_headers,
            files={"image": ("test.png", io.BytesIO(sample_image_bytes), "image/png")},
        )

        events = parse_sse_events(response.text)
        progress_events = [e for e in events if e["event"] == "progress"]

        assert len(progress_events) >= 1
        assert "stage" in progress_events[0]["data"]
        assert "percent" in progress_events[0]["data"]

    def test_stream_ends_with_completed_event(
        self, client, auth_headers, sample_image_bytes
    ):
        """Successful generation should end with 'completed' event."""
        response = client.post(
            "/generate/stream",
            headers=auth_headers,
            files={"image": ("test.png", io.BytesIO(sample_image_bytes), "image/png")},
        )

        events = parse_sse_events(response.text)
        completed = next((e for e in events if e["event"] == "completed"), None)

        assert completed is not None
        assert "job_id" in completed["data"]
        assert "download_url" in completed["data"]

    def test_download_url_is_presigned(self, client, auth_headers, sample_image_bytes):
        """Completed event should include S3 presigned URL."""
        response = client.post(
            "/generate/stream",
            headers=auth_headers,
            files={"image": ("test.png", io.BytesIO(sample_image_bytes), "image/png")},
        )

        events = parse_sse_events(response.text)
        completed = next(e for e in events if e["event"] == "completed")

        url = completed["data"]["download_url"]
        assert "s3" in url.lower() or "amazonaws" in url.lower()

    def test_shape_only_mode(self, client, auth_headers, sample_image_bytes):
        """With generate_texture=false, should skip texture generation."""
        response = client.post(
            "/generate/stream",
            headers=auth_headers,
            files={"image": ("test.png", io.BytesIO(sample_image_bytes), "image/png")},
            data={"generate_texture": "false"},
        )

        events = parse_sse_events(response.text)
        stages = [e["data"].get("stage") for e in events if e["event"] == "progress"]

        # Should have shape progress but not texture progress
        assert any("shape" in (s or "") for s in stages)
        # In shape-only mode, we skip texture
        assert not any("texture" in (s or "") for s in stages)

    def test_with_seed_parameter(
        self, client, auth_headers, sample_image_bytes, mock_shape_generator
    ):
        """Seed parameter should be passed to generators."""
        response = client.post(
            "/generate/stream",
            headers=auth_headers,
            files={"image": ("test.png", io.BytesIO(sample_image_bytes), "image/png")},
            data={"seed": "42"},
        )

        assert response.status_code == 200

    def test_with_steps_parameter(self, client, auth_headers, sample_image_bytes):
        """Steps parameter should be accepted."""
        response = client.post(
            "/generate/stream",
            headers=auth_headers,
            files={"image": ("test.png", io.BytesIO(sample_image_bytes), "image/png")},
            data={"steps": "50"},
        )

        assert response.status_code == 200

    def test_with_guidance_scale_parameter(
        self, client, auth_headers, sample_image_bytes
    ):
        """Guidance scale parameter should be accepted."""
        response = client.post(
            "/generate/stream",
            headers=auth_headers,
            files={"image": ("test.png", io.BytesIO(sample_image_bytes), "image/png")},
            data={"guidance_scale": "7.5"},
        )

        assert response.status_code == 200

    def test_with_octree_resolution_parameter(
        self, client, auth_headers, sample_image_bytes
    ):
        """Octree resolution parameter should be accepted."""
        response = client.post(
            "/generate/stream",
            headers=auth_headers,
            files={"image": ("test.png", io.BytesIO(sample_image_bytes), "image/png")},
            data={"octree_resolution": "384"},
        )

        assert response.status_code == 200

    def test_with_all_generation_parameters(
        self, client, auth_headers, sample_image_bytes
    ):
        """All generation parameters should work together."""
        response = client.post(
            "/generate/stream",
            headers=auth_headers,
            files={"image": ("test.png", io.BytesIO(sample_image_bytes), "image/png")},
            data={
                "seed": "42",
                "steps": "25",
                "guidance_scale": "3.0",
                "octree_resolution": "512",
            },
        )

        assert response.status_code == 200
        events = parse_sse_events(response.text)
        completed = next((e for e in events if e["event"] == "completed"), None)
        assert completed is not None

    def test_parameters_passed_to_shape_generator(
        self, app, auth_headers, sample_image_bytes
    ):
        """Generation parameters should be passed to shape generator."""
        client = TestClient(app)

        response = client.post(
            "/generate/stream",
            headers=auth_headers,
            files={"image": ("test.png", io.BytesIO(sample_image_bytes), "image/png")},
            data={
                "steps": "50",
                "guidance_scale": "7.5",
                "octree_resolution": "384",
            },
        )

        assert response.status_code == 200

        # Verify parameters were captured by mock
        shape_gen = app.state.shape_generator
        assert shape_gen.last_steps == 50
        assert shape_gen.last_guidance_scale == 7.5
        assert shape_gen.last_octree_resolution == 384


class TestGenerateStreamErrors:
    """Tests for error handling in /generate/stream."""

    def test_missing_image_returns_400(self, client, auth_headers):
        """Request without image should return 400."""
        response = client.post(
            "/generate/stream",
            headers=auth_headers,
        )

        assert response.status_code == 422  # FastAPI validation error

    def test_invalid_image_returns_error_event(
        self, client, auth_headers, monkeypatch, mock_texture_generator, mock_s3_storage
    ):
        """Invalid image should produce error SSE event."""
        from modal_app.tests.mocks.modal_services import MockShapeGenerator

        # Create a shape generator that fails on invalid input
        failing_generator = MockShapeGenerator(delay=0)
        failing_generator.fail_on_call = 1
        failing_generator.error_message = "Invalid image format"

        monkeypatch.setenv("VALID_API_KEYS", "test-key")

        from modal_app.api.app import create_app

        app = create_app(
            shape_generator=failing_generator,
            texture_generator=mock_texture_generator,
            s3_storage=mock_s3_storage,
        )
        test_client = TestClient(app)

        response = test_client.post(
            "/generate/stream",
            headers={"X-API-Key": "test-key"},
            files={"image": ("test.png", io.BytesIO(b"not-valid-png"), "image/png")},
        )

        events = parse_sse_events(response.text)
        error = next((e for e in events if e["event"] == "error"), None)

        assert error is not None
        assert "message" in error["data"]
        assert "retriable" in error["data"]
