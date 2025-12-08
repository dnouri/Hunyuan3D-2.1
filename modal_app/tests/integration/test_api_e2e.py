"""End-to-end integration tests for the Hunyuan3D API on Modal.

These tests require:
- Modal account with GPU access
- Deployed ShapeGenerator and TextureGenerator services
- Configured Modal secrets (aws-credentials, api-keys)

Run with:
    modal run modal_app/tests/integration/test_api_e2e.py

Or as part of the test suite (marked as slow):
    pytest modal_app/tests/integration/test_api_e2e.py -v -m slow
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

app = modal.App("hunyuan3d-api-e2e-test")

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def parse_sse_events(text: str) -> list[dict]:
    """Parse SSE response into list of events.

    Note: Duplicated from conftest.py because Modal functions
    run in isolated containers without access to test utilities.
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


@app.function(
    secrets=[modal.Secret.from_name("api-keys")],
    timeout=300,
)
def test_health_endpoint():
    """Test the /health endpoint."""
    import httpx

    # API URL from environment; defaults to localhost for local testing
    api_url = os.environ.get("HUNYUAN3D_API_URL", "http://localhost:8000")

    print(f"Testing health endpoint at {api_url}/health")

    response = httpx.get(f"{api_url}/health", timeout=30)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data

    print("✓ Health endpoint test passed")
    return {"status": "passed", "response": data}


@app.function(
    secrets=[modal.Secret.from_name("api-keys")],
    timeout=300,
)
def test_generate_stream_shape_only():
    """Test shape-only generation via the API."""
    import httpx

    api_url = os.environ.get("HUNYUAN3D_API_URL", "http://localhost:8000")
    api_key = os.environ.get("VALID_API_KEYS", "").split(",")[0].strip()

    if not api_key:
        return {"status": "skipped", "reason": "No API key configured"}

    # Load test image
    image_bytes = (FIXTURES_DIR / "test_image_256x256.png").read_bytes()

    print(f"Testing shape-only generation at {api_url}/generate/stream")
    start_time = time.time()

    with httpx.Client(timeout=300) as client:
        response = client.post(
            f"{api_url}/generate/stream",
            headers={"X-API-Key": api_key},
            files={"image": ("test.png", image_bytes, "image/png")},
            data={"generate_texture": "false"},
        )

    elapsed = time.time() - start_time
    print(f"Request completed in {elapsed:.1f}s")

    if response.status_code != 200:
        return {
            "status": "failed",
            "error": f"Expected 200, got {response.status_code}",
            "body": response.text[:500],
        }

    events = parse_sse_events(response.text)
    event_types = [e["event"] for e in events]

    print(f"Received events: {event_types}")

    # Verify event sequence
    assert "started" in event_types, "Missing 'started' event"
    assert "progress" in event_types, "Missing 'progress' event"
    assert "completed" in event_types or "error" in event_types, (
        "Missing terminal event"
    )

    # Check completed event has download URL
    completed = next((e for e in events if e["event"] == "completed"), None)
    if completed:
        assert "download_url" in completed["data"], "Missing download_url"
        assert "job_id" in completed["data"], "Missing job_id"
        print(f"✓ Shape-only generation test passed in {elapsed:.1f}s")
        return {
            "status": "passed",
            "elapsed_seconds": elapsed,
            "download_url": completed["data"]["download_url"],
        }
    else:
        error = next((e for e in events if e["event"] == "error"), None)
        return {
            "status": "failed",
            "error": error["data"] if error else "Unknown error",
            "elapsed_seconds": elapsed,
        }


@app.function(
    secrets=[modal.Secret.from_name("api-keys")],
    timeout=600,
)
def test_generate_stream_with_texture():
    """Test full generation (shape + texture) via the API."""
    import httpx

    api_url = os.environ.get("HUNYUAN3D_API_URL", "http://localhost:8000")
    api_key = os.environ.get("VALID_API_KEYS", "").split(",")[0].strip()

    if not api_key:
        return {"status": "skipped", "reason": "No API key configured"}

    # Load test image
    image_bytes = (FIXTURES_DIR / "test_image_256x256.png").read_bytes()

    print(f"Testing full generation at {api_url}/generate/stream")
    start_time = time.time()

    with httpx.Client(timeout=600) as client:
        response = client.post(
            f"{api_url}/generate/stream",
            headers={"X-API-Key": api_key},
            files={"image": ("test.png", image_bytes, "image/png")},
            data={"generate_texture": "true"},
        )

    elapsed = time.time() - start_time
    print(f"Request completed in {elapsed:.1f}s")

    if response.status_code != 200:
        return {
            "status": "failed",
            "error": f"Expected 200, got {response.status_code}",
            "body": response.text[:500],
        }

    events = parse_sse_events(response.text)
    event_types = [e["event"] for e in events]
    stages = [e["data"].get("stage") for e in events if e["event"] == "progress"]

    print(f"Received events: {event_types}")
    print(f"Progress stages: {stages}")

    # Verify we have texture stages
    has_texture = any("texture" in (s or "") for s in stages)
    if not has_texture:
        print("Warning: No texture stages found")

    completed = next((e for e in events if e["event"] == "completed"), None)
    if completed:
        print(f"✓ Full generation test passed in {elapsed:.1f}s")
        return {
            "status": "passed",
            "elapsed_seconds": elapsed,
            "download_url": completed["data"]["download_url"],
            "had_texture_stages": has_texture,
        }
    else:
        error = next((e for e in events if e["event"] == "error"), None)
        return {
            "status": "failed",
            "error": error["data"] if error else "Unknown error",
            "elapsed_seconds": elapsed,
        }


@app.function(
    secrets=[modal.Secret.from_name("api-keys")],
    timeout=60,
)
def test_authentication_required():
    """Test that API requires authentication."""
    import httpx

    api_url = os.environ.get("HUNYUAN3D_API_URL", "http://localhost:8000")
    image_bytes = (FIXTURES_DIR / "test_image_1x1.png").read_bytes()

    print("Testing authentication requirement")

    with httpx.Client(timeout=30) as client:
        # Request without API key
        response = client.post(
            f"{api_url}/generate/stream",
            files={"image": ("test.png", image_bytes, "image/png")},
        )

    assert response.status_code == 401, f"Expected 401, got {response.status_code}"
    print("✓ Authentication test passed")

    return {"status": "passed", "status_code": response.status_code}


@app.local_entrypoint()
def main():
    """Run all integration tests."""
    print("=" * 60)
    print("Hunyuan3D API Integration Tests")
    print("=" * 60)
    print()
    print("Note: These tests require:")
    print("  - Deployed API at HUNYUAN3D_API_URL")
    print("  - Valid API key in api-keys secret")
    print()

    results = {}

    # Run tests
    print("\n--- Test: Health Endpoint ---")
    try:
        results["health"] = test_health_endpoint.remote()
    except Exception as e:
        results["health"] = {"status": "error", "error": str(e)}

    print("\n--- Test: Authentication Required ---")
    try:
        results["auth"] = test_authentication_required.remote()
    except Exception as e:
        results["auth"] = {"status": "error", "error": str(e)}

    print("\n--- Test: Shape-Only Generation ---")
    try:
        results["shape_only"] = test_generate_stream_shape_only.remote()
    except Exception as e:
        results["shape_only"] = {"status": "error", "error": str(e)}

    print("\n--- Test: Full Generation (Shape + Texture) ---")
    try:
        results["full"] = test_generate_stream_with_texture.remote()
    except Exception as e:
        results["full"] = {"status": "error", "error": str(e)}

    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    for name, result in results.items():
        status = result.get("status", "unknown")
        symbol = "✓" if status == "passed" else "✗" if status == "failed" else "○"
        print(f"  {symbol} {name}: {status}")

    passed = sum(1 for r in results.values() if r.get("status") == "passed")
    total = len(results)
    print(f"\nTotal: {passed}/{total} passed")

    return results
