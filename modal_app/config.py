"""Centralized configuration for Modal deployment.

This module provides environment-aware configuration that supports both
local testing (with mocks) and Modal deployment (with real services).
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModalConfig:
    """Configuration for Modal deployment."""

    app_name: str = "hunyuan3d"

    shape_gpu: str = "A10G"
    shape_timeout: int = 300

    texture_gpu: str = "L40S"
    texture_timeout: int = 600

    api_timeout: int = 600

    model_base_path: str = "/models"
    app_path: str = "/app"


@dataclass(frozen=True)
class S3Config:
    """Configuration for S3 artifact storage."""

    bucket_name: str = ""
    region: str = "us-east-1"
    presigned_url_expiration: int = 3600
    result_prefix: str = "results/"

    @classmethod
    def from_env(cls) -> "S3Config":
        """Load configuration from environment variables."""
        return cls(
            bucket_name=os.environ.get("S3_BUCKET_NAME", ""),
            region=os.environ.get("AWS_REGION", "us-east-1"),
        )


def is_modal_environment() -> bool:
    """Check if running within a Modal container."""
    return os.environ.get("MODAL_ENVIRONMENT") is not None


def get_modal_config() -> ModalConfig:
    """Get the Modal configuration."""
    return ModalConfig()


def get_s3_config() -> S3Config:
    """Get the S3 configuration from environment."""
    return S3Config.from_env()
