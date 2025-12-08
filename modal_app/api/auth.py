"""API key authentication for the Hunyuan3D API.

Provides FastAPI dependencies for validating API keys passed via X-API-Key header.
Keys are validated against the VALID_API_KEYS environment variable (comma-separated).

Usage:
    from modal_app.api.auth import verify_api_key

    @app.get("/protected")
    async def protected_route(api_key: str = verify_api_key):
        return {"authenticated": True}
"""

from __future__ import annotations

import os
import secrets

from fastapi import Depends, Header, HTTPException


def _get_valid_api_keys() -> set[str]:
    """Get the set of valid API keys from environment.

    Returns:
        Set of valid API key strings

    Raises:
        HTTPException: 500 if VALID_API_KEYS is not configured
    """
    keys_str = os.environ.get("VALID_API_KEYS", "")
    if not keys_str:
        raise HTTPException(
            status_code=500,
            detail="API keys not configured. Set VALID_API_KEYS environment variable.",
        )
    return {key.strip() for key in keys_str.split(",") if key.strip()}


def _verify_api_key(x_api_key: str | None = Header(None)) -> str:
    """Validate the X-API-Key header against configured keys.

    Args:
        x_api_key: API key from X-API-Key header

    Returns:
        The validated API key string

    Raises:
        HTTPException: 401 if key is missing or invalid, 500 if not configured
    """
    if x_api_key is None:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide X-API-Key header.",
        )

    valid_keys = _get_valid_api_keys()

    # Use constant-time comparison to prevent timing attacks
    key_valid = any(
        secrets.compare_digest(x_api_key, valid_key) for valid_key in valid_keys
    )

    if not key_valid:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
        )

    return x_api_key


# FastAPI dependency - use this in route definitions
verify_api_key = Depends(_verify_api_key)
