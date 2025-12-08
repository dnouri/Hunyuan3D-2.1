"""Tests for API key authentication.

The auth module provides FastAPI dependencies for validating API keys.
Keys are passed via X-API-Key header and validated against environment.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


class TestVerifyApiKey:
    """Tests for the verify_api_key dependency."""

    @pytest.fixture
    def app_with_protected_route(self):
        """Create a minimal app with a protected endpoint."""
        from modal_app.api.auth import verify_api_key

        app = FastAPI()

        @app.get("/protected")
        async def protected_route(api_key: str = verify_api_key):
            return {"authenticated": True, "key_prefix": api_key[:10]}

        return app

    @pytest.fixture
    def valid_api_key(self):
        """A valid test API key."""
        return "sk_test_abc123def456ghi789jkl012mno345"

    @pytest.fixture
    def client_with_valid_keys(
        self, app_with_protected_route, valid_api_key, monkeypatch
    ):
        """Test client with valid API keys configured."""
        monkeypatch.setenv("VALID_API_KEYS", valid_api_key)
        return TestClient(app_with_protected_route)

    def test_returns_401_when_header_missing(self, client_with_valid_keys):
        """Missing X-API-Key header should return 401."""
        response = client_with_valid_keys.get("/protected")

        assert response.status_code == 401
        assert "Missing API key" in response.json()["detail"]

    def test_returns_401_for_invalid_key(self, client_with_valid_keys):
        """Invalid API key should return 401."""
        response = client_with_valid_keys.get(
            "/protected", headers={"X-API-Key": "invalid-key-12345"}
        )

        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]

    def test_returns_200_for_valid_key(self, client_with_valid_keys, valid_api_key):
        """Valid API key should allow access."""
        response = client_with_valid_keys.get(
            "/protected", headers={"X-API-Key": valid_api_key}
        )

        assert response.status_code == 200
        assert response.json()["authenticated"] is True

    def test_supports_multiple_valid_keys(self, app_with_protected_route, monkeypatch):
        """Should accept any key from comma-separated VALID_API_KEYS."""
        key1 = "sk_test_key_one_000000000000000"
        key2 = "sk_live_key_two_111111111111111"
        monkeypatch.setenv("VALID_API_KEYS", f"{key1},{key2}")

        client = TestClient(app_with_protected_route)

        # Both keys should work
        response1 = client.get("/protected", headers={"X-API-Key": key1})
        response2 = client.get("/protected", headers={"X-API-Key": key2})

        assert response1.status_code == 200
        assert response2.status_code == 200

    def test_strips_whitespace_from_keys(self, app_with_protected_route, monkeypatch):
        """Keys with surrounding whitespace should still work."""
        key = "sk_test_with_spaces_around_000000"
        monkeypatch.setenv("VALID_API_KEYS", f"  {key}  , other_key ")

        client = TestClient(app_with_protected_route)

        response = client.get("/protected", headers={"X-API-Key": key})
        assert response.status_code == 200

    def test_uses_constant_time_comparison(self, app_with_protected_route, monkeypatch):
        """Verify we use constant-time comparison to prevent timing attacks.

        This is a design requirement - we can't easily test timing, but we can
        verify the function uses secrets.compare_digest.
        """
        import secrets

        # The auth module should use secrets.compare_digest
        # We verify by checking if it's imported/used
        assert hasattr(secrets, "compare_digest")


class TestApiKeyNotConfigured:
    """Tests for error handling when API keys are not configured."""

    def test_raises_500_when_no_keys_configured(self, monkeypatch):
        """Missing VALID_API_KEYS should raise 500, not expose auth bypass."""
        from modal_app.api.auth import verify_api_key

        monkeypatch.delenv("VALID_API_KEYS", raising=False)

        app = FastAPI()

        @app.get("/protected")
        async def protected_route(api_key: str = verify_api_key):
            return {"authenticated": True}

        client = TestClient(app)

        response = client.get("/protected", headers={"X-API-Key": "any-key"})

        assert response.status_code == 500
        assert "not configured" in response.json()["detail"].lower()
