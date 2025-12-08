"""S3 storage module for artifact upload and presigned URL generation.

This module provides functions to upload GLB mesh files to S3 and generate
presigned URLs for secure, time-limited downloads.

Configuration via environment variables:
    AWS_ACCESS_KEY_ID: AWS access key
    AWS_SECRET_ACCESS_KEY: AWS secret key
    AWS_REGION: AWS region (default: us-east-1)
    S3_BUCKET_NAME: Target bucket name

Interface:
    upload_artifact(job_id, data, content_type) -> presigned_url
    generate_presigned_url(job_id, expires_in) -> presigned_url
"""

from __future__ import annotations

import os

import boto3
from botocore.config import Config


def _get_bucket_name() -> str:
    """Get the S3 bucket name from environment.

    Returns:
        Bucket name string

    Raises:
        RuntimeError: If S3_BUCKET_NAME is not configured
    """
    bucket = os.environ.get("S3_BUCKET_NAME")
    if not bucket:
        raise RuntimeError(
            "S3_BUCKET_NAME environment variable is not set. "
            "Configure it via Modal secrets."
        )
    return bucket


def _get_s3_client():
    """Create an S3 client with configuration from environment.

    Returns:
        boto3 S3 client

    Raises:
        RuntimeError: If S3_BUCKET_NAME is not configured
    """
    # Validate bucket is configured (fail fast)
    _get_bucket_name()

    region = os.environ.get("AWS_REGION", "us-east-1")
    config = Config(signature_version="s3v4")

    return boto3.client("s3", region_name=region, config=config)


def _get_object_key(job_id: str) -> str:
    """Build the S3 object key for a job.

    Args:
        job_id: Unique identifier for the job

    Returns:
        S3 key in format: results/{job_id}.glb
    """
    return f"results/{job_id}.glb"


def upload_artifact(job_id: str, data: bytes, content_type: str) -> str:
    """Upload artifact to S3 and return presigned URL.

    Args:
        job_id: Unique identifier for the job (used in S3 key)
        data: Binary data to upload
        content_type: MIME type for the content (e.g., "model/gltf-binary")

    Returns:
        Presigned URL for downloading the uploaded artifact

    Raises:
        ValueError: If job_id or data is empty
        RuntimeError: If S3_BUCKET_NAME is not configured
    """
    if not job_id:
        raise ValueError("job_id cannot be empty")
    if not data:
        raise ValueError("data cannot be empty")

    s3 = _get_s3_client()
    bucket = _get_bucket_name()
    key = _get_object_key(job_id)

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=data,
        ContentType=content_type,
        ContentDisposition=f'attachment; filename="{job_id}.glb"',
    )

    return generate_presigned_url(job_id)


def generate_presigned_url(job_id: str, expires_in: int = 3600) -> str:
    """Generate a presigned URL for an existing artifact.

    Args:
        job_id: Unique identifier for the job
        expires_in: URL expiration time in seconds (default: 1 hour)

    Returns:
        Presigned URL for downloading the artifact

    Raises:
        ValueError: If job_id is empty

    Note:
        This generates a URL without checking if the object exists.
        If the object doesn't exist, the download will fail with 404.
    """
    if not job_id:
        raise ValueError("job_id cannot be empty")

    s3 = _get_s3_client()
    bucket = _get_bucket_name()
    key = _get_object_key(job_id)

    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_in,
    )

    return url
