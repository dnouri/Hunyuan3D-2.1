"""Shared pytest fixtures for Modal integration tests.

Provides common fixtures including mock services and test data.
Mocks replicate the interface and behavior of real Modal services.
"""

import json
from pathlib import Path

import pytest


def parse_sse_events(text: str) -> list[dict]:
    """Parse SSE response text into list of event dictionaries.

    Args:
        text: Raw SSE response text

    Returns:
        List of {"event": str, "data": dict} dictionaries
    """
    events = []
    current_event = {}

    for line in text.split("\n"):
        if line.startswith("event: "):
            current_event["event"] = line[7:]
        elif line.startswith("data: "):
            try:
                current_event["data"] = json.loads(line[6:])
            except json.JSONDecodeError:
                current_event["data"] = {"raw": line[6:]}
        elif line == "" and current_event:
            if "event" in current_event:
                events.append(current_event)
            current_event = {}

    return events


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Small valid PNG image for testing (1x1 pixel)."""
    return (FIXTURES_DIR / "test_image_1x1.png").read_bytes()


@pytest.fixture
def sample_image_256() -> bytes:
    """Larger valid PNG image for testing (256x256 pixels)."""
    return (FIXTURES_DIR / "test_image_256x256.png").read_bytes()


@pytest.fixture
def sample_glb_bytes() -> bytes:
    """Valid GLB mesh file for testing."""
    return (FIXTURES_DIR / "test_mesh.glb").read_bytes()


@pytest.fixture
def mock_shape_generator():
    """Mock ShapeGenerator with no delay for fast unit tests."""
    from modal_app.tests.mocks.modal_services import MockShapeGenerator

    return MockShapeGenerator(delay=0)


@pytest.fixture
def mock_texture_generator():
    """Mock TextureGenerator with no delay for fast unit tests."""
    from modal_app.tests.mocks.modal_services import MockTextureGenerator

    return MockTextureGenerator(delay=0)


@pytest.fixture
def mock_s3_storage():
    """Mock S3 storage for testing without AWS."""
    from modal_app.tests.mocks.modal_services import MockS3Storage

    return MockS3Storage()


@pytest.fixture
def job_id() -> str:
    """Sample job ID for testing."""
    return "test-job-12345678"


@pytest.fixture
def s3_bucket(monkeypatch):
    """Configure environment for moto-mocked S3 and create bucket.

    Sets up:
    - AWS credentials (fake)
    - S3_BUCKET_NAME environment variable
    - Creates the test bucket in moto's mock S3

    IMPORTANT: Use with @mock_aws decorator on the test. The decorator must
    be applied to the test function, not the fixture.
    """
    import boto3
    from moto import mock_aws

    # Set environment variables for the S3 module
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("S3_BUCKET_NAME", "test-bucket")

    # Start moto mock and create bucket
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        yield "test-bucket"
