"""Integration test for ShapeGenerator service on Modal.

This test runs the ShapeGenerator with real GPU to verify:
- Pipeline initialization with memory snapshots
- Shape generation from input image
- Output is valid GLB mesh

Run with:
    modal run tests/integration/test_shape_generator_modal.py

Expected:
- First run: ~60-90s (cold start + inference)
- Subsequent runs: ~30-45s (warm start + inference)

Generation Parameters:
    The shape generator accepts optional parameters (with defaults):
    - seed: int | None (reproducibility)
    - steps: int = 30 (inference steps, higher = better quality)
    - guidance_scale: float = 5.0 (classifier-free guidance)
    - octree_resolution: int = 256 (mesh detail, higher = more detail)
"""

from pathlib import Path

from modal_app.services.shape_generator import (
    ShapeGenerator,
    app,
    shape_generator_image,
)


@app.function(image=shape_generator_image)
def run_integration_test() -> dict:
    """Run shape generation integration test.

    Returns:
        dict with test results including:
        - success: bool
        - output_size: int (bytes)
        - inference_time: float (seconds)
        - errors: list of error messages
    """
    import time

    results = {
        "success": False,
        "output_size": 0,
        "inference_time": 0.0,
        "errors": [],
        "checks": {},
    }

    try:
        # The demo.png is at /app/assets/demo.png in the container
        image_path = Path("/app/assets/demo.png")
        if not image_path.exists():
            results["errors"].append(f"Test image not found: {image_path}")
            return results

        image_bytes = image_path.read_bytes()
        results["checks"]["image_loaded"] = True
        results["input_size"] = len(image_bytes)

        # Call the shape generator remotely (on GPU container)
        # Time the generation
        start_time = time.time()
        output_bytes = ShapeGenerator().generate.remote(image_bytes, seed=42)
        inference_time = time.time() - start_time

        results["inference_time"] = round(inference_time, 2)
        results["output_size"] = len(output_bytes)
        results["checks"]["generation_completed"] = True

        # Validate output is GLB
        # GLB files start with magic bytes: glTF
        if output_bytes[:4] == b"glTF":
            results["checks"]["valid_glb_header"] = True
        else:
            results["checks"]["valid_glb_header"] = False
            results["errors"].append("Output does not have valid GLB header")

        # Check output size is reasonable (GLB should be >1KB, <100MB)
        if 1024 < len(output_bytes) < 100 * 1024 * 1024:
            results["checks"]["reasonable_size"] = True
        else:
            results["checks"]["reasonable_size"] = False
            results["errors"].append(f"Output size {len(output_bytes)} is unusual")

        # All checks passed
        if all(results["checks"].values()):
            results["success"] = True

    except Exception as e:
        import traceback

        results["errors"].append(f"{type(e).__name__}: {str(e)[:200]}")
        results["traceback"] = traceback.format_exc()

    return results


@app.local_entrypoint()
def main():
    """Run the integration test and print results."""
    print("=" * 70)
    print("SHAPE GENERATOR INTEGRATION TEST")
    print("=" * 70)
    print("\nRunning shape generation on Modal GPU...")
    print("(First run may take 2-3 minutes for container + model initialization)")
    print()

    result = run_integration_test.remote()

    print(f"{'=' * 70}")
    print("TEST RESULTS")
    print(f"{'=' * 70}")

    print(f"\nInput size: {result.get('input_size', 'N/A')} bytes")
    print(f"Output size: {result.get('output_size', 'N/A')} bytes")
    print(f"Inference time: {result.get('inference_time', 'N/A')}s")

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
        print("INTEGRATION TEST PASSED")
    else:
        print("INTEGRATION TEST FAILED")
    print(f"{'=' * 70}")

    return result.get("success", False)
