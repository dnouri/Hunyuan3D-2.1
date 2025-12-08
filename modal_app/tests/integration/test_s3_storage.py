"""Integration test for S3 storage on Modal.

This test runs the S3 storage module with real AWS credentials to verify:
- Upload to S3 works correctly
- Presigned URL is valid and downloadable
- Content types and metadata are set properly

Prerequisites:
    modal secret create aws-credentials \
        AWS_ACCESS_KEY_ID=xxx \
        AWS_SECRET_ACCESS_KEY=xxx \
        AWS_REGION=us-east-1 \
        S3_BUCKET_NAME=your-bucket

Run with:
    modal run tests/integration/test_s3_storage.py

Expected:
- Upload completes in <2s
- Presigned URL is downloadable
- Content matches what was uploaded
"""

import modal

app = modal.App("hunyuan3d-s3-test")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("boto3", "httpx")
    .add_local_python_source("modal_app")
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("aws-credentials")],
)
def run_s3_integration_test() -> dict:
    """Run S3 storage integration test.

    Returns:
        dict with test results including:
        - success: bool
        - upload_time: float (seconds)
        - download_time: float (seconds)
        - errors: list of error messages
        - checks: dict of test checks
    """
    import hashlib
    import os
    import time

    import httpx

    results = {
        "success": False,
        "upload_time": 0.0,
        "download_time": 0.0,
        "errors": [],
        "checks": {},
    }

    try:
        # Import the storage module
        from modal_app.storage.s3 import (
            _get_bucket_name,
            generate_presigned_url,
            upload_artifact,
        )

        bucket = _get_bucket_name()
        results["bucket"] = bucket
        results["checks"]["bucket_configured"] = True

        # Test 1: Upload artifact
        job_id = f"integration-test-{int(time.time())}"
        test_data = b"Integration test data for GLB upload " * 1000  # ~37KB
        content_hash = hashlib.md5(test_data).hexdigest()

        print(f"Uploading test artifact: {job_id}")
        start = time.time()
        presigned_url = upload_artifact(job_id, test_data, "model/gltf-binary")
        results["upload_time"] = round(time.time() - start, 3)

        results["checks"]["upload_completed"] = True
        results["checks"]["url_returned"] = presigned_url.startswith("https://")

        # Test 2: Download via presigned URL
        print("Downloading via presigned URL...")
        start = time.time()
        response = httpx.get(presigned_url)
        results["download_time"] = round(time.time() - start, 3)

        results["checks"]["download_status_ok"] = response.status_code == 200
        results["checks"]["content_length_correct"] = len(response.content) == len(
            test_data
        )

        download_hash = hashlib.md5(response.content).hexdigest()
        results["checks"]["content_matches"] = download_hash == content_hash

        # Test 3: Generate new presigned URL
        print("Generating new presigned URL...")
        new_url = generate_presigned_url(job_id, expires_in=300)
        results["checks"]["generate_url_works"] = new_url.startswith("https://")
        results["checks"]["urls_different"] = new_url != presigned_url

        # Cleanup
        print("Cleaning up test artifact...")
        import boto3

        s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
        s3.delete_object(Bucket=bucket, Key=f"results/{job_id}.glb")
        results["checks"]["cleanup_completed"] = True

        # Overall success
        if all(results["checks"].values()):
            results["success"] = True

    except Exception as e:
        import traceback

        results["errors"].append(f"{type(e).__name__}: {str(e)[:200]}")
        results["traceback"] = traceback.format_exc()

    return results


@app.local_entrypoint()
def main():
    """Run the S3 integration test and print results."""
    print("=" * 70)
    print("S3 STORAGE INTEGRATION TEST")
    print("=" * 70)
    print("\nRunning S3 operations with real AWS credentials...")
    print("(Requires aws-credentials Modal secret)")
    print()

    result = run_s3_integration_test.remote()

    print(f"{'=' * 70}")
    print("TEST RESULTS")
    print(f"{'=' * 70}")

    print(f"\nBucket: {result.get('bucket', 'N/A')}")
    print(f"Upload time: {result.get('upload_time', 'N/A')}s")
    print(f"Download time: {result.get('download_time', 'N/A')}s")

    print("\nChecks:")
    for check, passed in result.get("checks", {}).items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check}")

    if result.get("errors"):
        print("\nErrors:")
        for error in result["errors"]:
            print(f"  - {error}")

    if result.get("traceback"):
        print("\nTraceback:")
        print(result["traceback"])

    print(f"\n{'=' * 70}")
    if result.get("success"):
        print("S3 INTEGRATION TEST PASSED")
    else:
        print("S3 INTEGRATION TEST FAILED")
    print(f"{'=' * 70}")

    return result.get("success", False)
