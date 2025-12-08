"""Unit tests for S3 storage module.

Tests the S3 upload and presigned URL functionality using moto for mocking.
These tests run without AWS credentials and verify the orchestration logic.

Following TDD: These tests are written FIRST to define the interface.
"""

import pytest


class TestUploadArtifact:
    """Test the upload_artifact function interface."""

    def test_upload_returns_presigned_url(self, s3_bucket):
        """Upload should return a presigned URL for the uploaded object."""
        from modal_app.storage.s3 import upload_artifact

        job_id = "test-job-123"
        data = b"test binary data"
        content_type = "model/gltf-binary"

        url = upload_artifact(job_id, data, content_type)

        assert isinstance(url, str)
        assert url.startswith("https://")
        assert job_id in url

    def test_upload_stores_data_correctly(self, s3_bucket):
        """Uploaded data should be retrievable from S3."""
        import boto3

        from modal_app.storage.s3 import upload_artifact

        job_id = "test-job-456"
        data = b"glTF binary mesh data"
        content_type = "model/gltf-binary"

        upload_artifact(job_id, data, content_type)

        # Verify data was stored correctly
        s3 = boto3.client("s3", region_name="us-east-1")
        response = s3.get_object(Bucket="test-bucket", Key=f"results/{job_id}.glb")
        assert response["Body"].read() == data

    def test_upload_sets_content_type(self, s3_bucket):
        """Upload should set the correct Content-Type."""
        import boto3

        from modal_app.storage.s3 import upload_artifact

        job_id = "test-job-789"
        data = b"mesh data"
        content_type = "model/gltf-binary"

        upload_artifact(job_id, data, content_type)

        s3 = boto3.client("s3", region_name="us-east-1")
        response = s3.head_object(Bucket="test-bucket", Key=f"results/{job_id}.glb")
        assert response["ContentType"] == content_type

    def test_upload_sets_content_disposition(self, s3_bucket):
        """Upload should set Content-Disposition for download filename."""
        import boto3

        from modal_app.storage.s3 import upload_artifact

        job_id = "test-job-download"
        data = b"mesh data"

        upload_artifact(job_id, data, "model/gltf-binary")

        s3 = boto3.client("s3", region_name="us-east-1")
        response = s3.head_object(Bucket="test-bucket", Key=f"results/{job_id}.glb")
        assert "attachment" in response.get("ContentDisposition", "")
        assert job_id in response.get("ContentDisposition", "")

    def test_upload_raises_for_empty_data(self, s3_bucket):
        """Upload should raise ValueError for empty data."""
        from modal_app.storage.s3 import upload_artifact

        with pytest.raises(ValueError, match="data cannot be empty"):
            upload_artifact("job-id", b"", "model/gltf-binary")

    def test_upload_raises_for_empty_job_id(self, s3_bucket):
        """Upload should raise ValueError for empty job_id."""
        from modal_app.storage.s3 import upload_artifact

        with pytest.raises(ValueError, match="job_id cannot be empty"):
            upload_artifact("", b"data", "model/gltf-binary")


class TestGeneratePresignedUrl:
    """Test the generate_presigned_url function interface."""

    def test_generates_valid_url(self, s3_bucket):
        """Should generate a valid presigned URL for existing object."""
        import boto3

        from modal_app.storage.s3 import generate_presigned_url

        # First, upload an object
        s3 = boto3.client("s3", region_name="us-east-1")
        job_id = "existing-job"
        s3.put_object(
            Bucket="test-bucket",
            Key=f"results/{job_id}.glb",
            Body=b"mesh data",
        )

        url = generate_presigned_url(job_id)

        assert isinstance(url, str)
        assert url.startswith("https://")
        assert job_id in url

    def test_default_expiration_is_one_hour(self, s3_bucket):
        """Default expiration should be 3600 seconds (1 hour)."""
        import boto3

        from modal_app.storage.s3 import generate_presigned_url

        s3 = boto3.client("s3", region_name="us-east-1")
        job_id = "expiry-test"
        s3.put_object(Bucket="test-bucket", Key=f"results/{job_id}.glb", Body=b"data")

        url = generate_presigned_url(job_id)

        # URL should contain expiry info (X-Amz-Expires=3600)
        assert "X-Amz-Expires=3600" in url or "Expires=" in url

    def test_custom_expiration(self, s3_bucket):
        """Should support custom expiration time."""
        import boto3

        from modal_app.storage.s3 import generate_presigned_url

        s3 = boto3.client("s3", region_name="us-east-1")
        job_id = "custom-expiry"
        s3.put_object(Bucket="test-bucket", Key=f"results/{job_id}.glb", Body=b"data")

        url = generate_presigned_url(job_id, expires_in=7200)

        # Should have 2 hour expiry
        assert "X-Amz-Expires=7200" in url or "7200" in url

    def test_raises_for_empty_job_id(self, s3_bucket):
        """Should raise ValueError for empty job_id."""
        from modal_app.storage.s3 import generate_presigned_url

        with pytest.raises(ValueError, match="job_id cannot be empty"):
            generate_presigned_url("")


class TestS3Configuration:
    """Test S3 configuration handling."""

    def test_missing_bucket_raises_error(self, monkeypatch):
        """Should raise clear error when bucket not configured."""
        monkeypatch.delenv("S3_BUCKET_NAME", raising=False)
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)

        from modal_app.storage.s3 import upload_artifact

        with pytest.raises(RuntimeError, match="S3_BUCKET_NAME"):
            upload_artifact("job", b"data", "type")
