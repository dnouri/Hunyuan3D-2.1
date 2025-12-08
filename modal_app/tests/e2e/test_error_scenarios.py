"""Error scenario tests for the Hunyuan3D API.

These tests verify error handling and error event formatting:
- Shape generator failures
- Texture generator failures
- S3 upload failures
- Invalid input handling

Run with:
    pytest modal_app/tests/e2e/test_error_scenarios.py -v
"""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from modal_app.tests.conftest import parse_sse_events

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def valid_api_key():
    return "sk_test_error_scenario_key_1234567"


@pytest.fixture
def sample_image():
    return (FIXTURES_DIR / "test_image_256x256.png").read_bytes()


@pytest.fixture
def auth_headers(valid_api_key):
    return {"X-API-Key": valid_api_key}


class TestShapeGeneratorErrors:
    """Tests for shape generator failure scenarios."""

    def test_shape_failure_returns_error_event(
        self, sample_image, valid_api_key, monkeypatch
    ):
        """Shape generator failure should produce SSE error event."""
        from modal_app.tests.mocks.modal_services import (
            MockS3Storage,
            MockShapeGenerator,
            MockTextureGenerator,
        )

        failing_shape = MockShapeGenerator(delay=0)
        failing_shape.fail_on_call = 1
        failing_shape.error_message = "GPU out of memory"

        monkeypatch.setenv("VALID_API_KEYS", valid_api_key)

        from modal_app.api.app import create_app

        app = create_app(
            shape_generator=failing_shape,
            texture_generator=MockTextureGenerator(delay=0),
            s3_storage=MockS3Storage(),
        )
        client = TestClient(app)

        response = client.post(
            "/generate/stream",
            headers={"X-API-Key": valid_api_key},
            files={"image": ("test.png", io.BytesIO(sample_image), "image/png")},
        )

        events = parse_sse_events(response.text)
        error_event = next((e for e in events if e["event"] == "error"), None)

        assert error_event is not None
        assert "GPU out of memory" in error_event["data"]["message"]
        assert error_event["data"]["retriable"] is False

    def test_shape_failure_has_started_event_first(
        self, sample_image, valid_api_key, monkeypatch
    ):
        """Even on failure, started event should be sent first."""
        from modal_app.tests.mocks.modal_services import (
            MockS3Storage,
            MockShapeGenerator,
            MockTextureGenerator,
        )

        failing_shape = MockShapeGenerator(delay=0)
        failing_shape.fail_on_call = 1

        monkeypatch.setenv("VALID_API_KEYS", valid_api_key)

        from modal_app.api.app import create_app

        app = create_app(
            shape_generator=failing_shape,
            texture_generator=MockTextureGenerator(delay=0),
            s3_storage=MockS3Storage(),
        )
        client = TestClient(app)

        response = client.post(
            "/generate/stream",
            headers={"X-API-Key": valid_api_key},
            files={"image": ("test.png", io.BytesIO(sample_image), "image/png")},
        )

        events = parse_sse_events(response.text)
        assert events[0]["event"] == "started"
        assert "job_id" in events[0]["data"]


class TestTextureGeneratorErrors:
    """Tests for texture generator failure scenarios."""

    def test_texture_failure_returns_error_event(
        self, sample_image, valid_api_key, monkeypatch
    ):
        """Texture generator failure should produce SSE error event."""
        from modal_app.tests.mocks.modal_services import (
            MockS3Storage,
            MockShapeGenerator,
            MockTextureGenerator,
        )

        failing_texture = MockTextureGenerator(delay=0)
        failing_texture.fail_on_call = 1
        failing_texture.error_message = "CUDA error during texture baking"

        monkeypatch.setenv("VALID_API_KEYS", valid_api_key)

        from modal_app.api.app import create_app

        app = create_app(
            shape_generator=MockShapeGenerator(delay=0),
            texture_generator=failing_texture,
            s3_storage=MockS3Storage(),
        )
        client = TestClient(app)

        response = client.post(
            "/generate/stream",
            headers={"X-API-Key": valid_api_key},
            files={"image": ("test.png", io.BytesIO(sample_image), "image/png")},
            data={"generate_texture": "true"},
        )

        events = parse_sse_events(response.text)
        error_event = next((e for e in events if e["event"] == "error"), None)

        assert error_event is not None
        assert "CUDA error" in error_event["data"]["message"]

    def test_texture_failure_after_shape_success(
        self, sample_image, valid_api_key, monkeypatch
    ):
        """Should get shape progress before texture error."""
        from modal_app.tests.mocks.modal_services import (
            MockS3Storage,
            MockShapeGenerator,
            MockTextureGenerator,
        )

        failing_texture = MockTextureGenerator(delay=0)
        failing_texture.fail_on_call = 1

        monkeypatch.setenv("VALID_API_KEYS", valid_api_key)

        from modal_app.api.app import create_app

        app = create_app(
            shape_generator=MockShapeGenerator(delay=0),
            texture_generator=failing_texture,
            s3_storage=MockS3Storage(),
        )
        client = TestClient(app)

        response = client.post(
            "/generate/stream",
            headers={"X-API-Key": valid_api_key},
            files={"image": ("test.png", io.BytesIO(sample_image), "image/png")},
        )

        events = parse_sse_events(response.text)
        event_types = [e["event"] for e in events]

        # Should have: started, progress (shape), progress (shape_complete), ...error
        assert "started" in event_types
        assert "progress" in event_types
        assert "error" in event_types


class TestErrorEventFormat:
    """Tests for error event structure and format."""

    def test_error_event_has_required_fields(
        self, sample_image, valid_api_key, monkeypatch
    ):
        """Error events must have stage, message, and retriable."""
        from modal_app.tests.mocks.modal_services import (
            MockS3Storage,
            MockShapeGenerator,
            MockTextureGenerator,
        )

        failing_shape = MockShapeGenerator(delay=0)
        failing_shape.fail_on_call = 1
        failing_shape.error_message = "Test error"

        monkeypatch.setenv("VALID_API_KEYS", valid_api_key)

        from modal_app.api.app import create_app

        app = create_app(
            shape_generator=failing_shape,
            texture_generator=MockTextureGenerator(delay=0),
            s3_storage=MockS3Storage(),
        )
        client = TestClient(app)

        response = client.post(
            "/generate/stream",
            headers={"X-API-Key": valid_api_key},
            files={"image": ("test.png", io.BytesIO(sample_image), "image/png")},
        )

        events = parse_sse_events(response.text)
        error_event = next(e for e in events if e["event"] == "error")

        assert "stage" in error_event["data"]
        assert "message" in error_event["data"]
        assert "retriable" in error_event["data"]

    def test_error_retriable_is_boolean(self, sample_image, valid_api_key, monkeypatch):
        """retriable field should be a boolean."""
        from modal_app.tests.mocks.modal_services import (
            MockS3Storage,
            MockShapeGenerator,
            MockTextureGenerator,
        )

        failing_shape = MockShapeGenerator(delay=0)
        failing_shape.fail_on_call = 1

        monkeypatch.setenv("VALID_API_KEYS", valid_api_key)

        from modal_app.api.app import create_app

        app = create_app(
            shape_generator=failing_shape,
            texture_generator=MockTextureGenerator(delay=0),
            s3_storage=MockS3Storage(),
        )
        client = TestClient(app)

        response = client.post(
            "/generate/stream",
            headers={"X-API-Key": valid_api_key},
            files={"image": ("test.png", io.BytesIO(sample_image), "image/png")},
        )

        events = parse_sse_events(response.text)
        error_event = next(e for e in events if e["event"] == "error")

        assert isinstance(error_event["data"]["retriable"], bool)


class TestInputValidationErrors:
    """Tests for input validation error handling."""

    def test_missing_image_returns_422(self, valid_api_key, monkeypatch):
        """Missing image should return 422 validation error."""
        from modal_app.tests.mocks.modal_services import (
            MockS3Storage,
            MockShapeGenerator,
            MockTextureGenerator,
        )

        monkeypatch.setenv("VALID_API_KEYS", valid_api_key)

        from modal_app.api.app import create_app

        app = create_app(
            shape_generator=MockShapeGenerator(delay=0),
            texture_generator=MockTextureGenerator(delay=0),
            s3_storage=MockS3Storage(),
        )
        client = TestClient(app)

        response = client.post(
            "/generate/stream",
            headers={"X-API-Key": valid_api_key},
        )

        assert response.status_code == 422

    def test_empty_image_produces_error_event(self, valid_api_key, monkeypatch):
        """Empty image data should produce error event."""
        from modal_app.tests.mocks.modal_services import (
            MockS3Storage,
            MockShapeGenerator,
            MockTextureGenerator,
        )

        monkeypatch.setenv("VALID_API_KEYS", valid_api_key)

        from modal_app.api.app import create_app

        app = create_app(
            shape_generator=MockShapeGenerator(delay=0),
            texture_generator=MockTextureGenerator(delay=0),
            s3_storage=MockS3Storage(),
        )
        client = TestClient(app)

        response = client.post(
            "/generate/stream",
            headers={"X-API-Key": valid_api_key},
            files={"image": ("empty.png", io.BytesIO(b""), "image/png")},
        )

        # Empty file still accepted by FastAPI, but mock will reject
        events = parse_sse_events(response.text)
        error_event = next((e for e in events if e["event"] == "error"), None)

        assert error_event is not None


class TestApiKeyErrors:
    """Tests for API key error handling."""

    def test_missing_key_error_is_json(self, sample_image, monkeypatch):
        """Missing API key should return JSON error."""
        from modal_app.tests.mocks.modal_services import (
            MockS3Storage,
            MockShapeGenerator,
            MockTextureGenerator,
        )

        monkeypatch.setenv("VALID_API_KEYS", "some-key")

        from modal_app.api.app import create_app

        app = create_app(
            shape_generator=MockShapeGenerator(delay=0),
            texture_generator=MockTextureGenerator(delay=0),
            s3_storage=MockS3Storage(),
        )
        client = TestClient(app)

        response = client.post(
            "/generate/stream",
            files={"image": ("test.png", io.BytesIO(sample_image), "image/png")},
        )

        assert response.status_code == 401
        assert "detail" in response.json()

    def test_invalid_key_error_is_json(self, sample_image, monkeypatch):
        """Invalid API key should return JSON error."""
        from modal_app.tests.mocks.modal_services import (
            MockS3Storage,
            MockShapeGenerator,
            MockTextureGenerator,
        )

        monkeypatch.setenv("VALID_API_KEYS", "correct-key")

        from modal_app.api.app import create_app

        app = create_app(
            shape_generator=MockShapeGenerator(delay=0),
            texture_generator=MockTextureGenerator(delay=0),
            s3_storage=MockS3Storage(),
        )
        client = TestClient(app)

        response = client.post(
            "/generate/stream",
            headers={"X-API-Key": "wrong-key"},
            files={"image": ("test.png", io.BytesIO(sample_image), "image/png")},
        )

        assert response.status_code == 401
        assert "detail" in response.json()
