"""End-to-End integration test for ShapeGenerator -> TextureGenerator pipeline.

This test validates the full pipeline from image to textured mesh:
1. Calls deployed ShapeGenerator with input image
2. Passes output mesh to deployed TextureGenerator
3. Verifies final result is valid textured GLB

Run with:
    modal run tests/integration/test_pipeline_e2e.py

Prerequisites:
    Both services must be deployed first:
    - modal deploy modal_app/services/shape_generator.py
    - modal deploy modal_app/services/texture_generator.py

Expected:
- Total time: ~90s (shape: ~30s + texture: ~60s)
- Output: Valid GLB with PBR textures

Note:
    This test uses modal.Function.lookup() to call deployed services,
    testing the actual production API. This is the true end-to-end test
    that validates cross-service integration.
"""

from pathlib import Path

import modal

app = modal.App("hunyuan3d-e2e-test")


@app.local_entrypoint()
def main():
    """Run the E2E test using deployed services.

    This test calls the deployed ShapeGenerator and TextureGenerator services
    via modal.Function.lookup(), testing the actual production flow.
    """
    import time

    print("=" * 70)
    print("HUNYUAN3D END-TO-END PIPELINE TEST")
    print("=" * 70)
    print("\nRunning full pipeline: Image → Shape → Texture")
    print("(Using deployed services via modal.Function.lookup())")
    print()

    results = {
        "success": False,
        "stages": {},
        "errors": [],
        "checks": {},
    }

    try:
        # Look up deployed services
        print("Looking up deployed services...")
        try:
            ShapeGenerator = modal.Cls.lookup(
                "hunyuan3d-shape-generator", "ShapeGenerator"
            )
            TextureGenerator = modal.Cls.lookup(
                "hunyuan3d-texture-generator", "TextureGenerator"
            )
            results["checks"]["services_found"] = True
            print("  Found: ShapeGenerator, TextureGenerator")
        except modal.exception.NotFoundError as e:
            results["errors"].append(
                f"Services not deployed. Run 'modal deploy' first: {e}"
            )
            results["checks"]["services_found"] = False
            _print_results(results)
            return False

        # Load test image from local assets (not container)
        image_path = Path(__file__).parent.parent.parent.parent / "assets" / "demo.png"
        if not image_path.exists():
            # Try alternate location
            image_path = Path("assets/demo.png")
        if not image_path.exists():
            results["errors"].append(f"Test image not found: {image_path}")
            _print_results(results)
            return False

        image_bytes = image_path.read_bytes()
        results["stages"]["input"] = {
            "image_size": len(image_bytes),
        }
        print(f"\nLoaded test image: {len(image_bytes)} bytes")

        # Stage 1: Shape Generation
        print()
        print("=" * 50)
        print("Stage 1: Shape Generation (A10G GPU)")
        print("=" * 50)

        shape_start = time.time()
        mesh_bytes = ShapeGenerator().generate.remote(image_bytes, seed=42)
        shape_time = time.time() - shape_start

        results["stages"]["shape"] = {
            "time": round(shape_time, 2),
            "output_size": len(mesh_bytes),
            "valid_glb": mesh_bytes[:4] == b"glTF",
        }
        print(f"Shape generation: {shape_time:.2f}s, {len(mesh_bytes)} bytes")

        if mesh_bytes[:4] != b"glTF":
            results["errors"].append("Shape output is not valid GLB")
            results["checks"]["shape_completed"] = False
            _print_results(results)
            return False

        results["checks"]["shape_completed"] = True

        # Stage 2: Texture Generation
        print()
        print("=" * 50)
        print("Stage 2: Texture Generation (L40S GPU)")
        print("=" * 50)

        texture_start = time.time()
        textured_bytes = TextureGenerator().generate.remote(
            mesh_bytes,
            image_bytes,
            seed=42,
        )
        texture_time = time.time() - texture_start

        results["stages"]["texture"] = {
            "time": round(texture_time, 2),
            "output_size": len(textured_bytes),
            "valid_glb": textured_bytes[:4] == b"glTF",
        }
        print(f"Texture generation: {texture_time:.2f}s, {len(textured_bytes)} bytes")

        # Validation checks
        results["checks"]["texture_completed"] = True
        results["checks"]["output_is_glb"] = textured_bytes[:4] == b"glTF"
        results["checks"]["texture_larger_than_mesh"] = len(textured_bytes) > len(
            mesh_bytes
        )

        # Calculate totals
        total_time = shape_time + texture_time
        results["stages"]["total"] = {
            "time": round(total_time, 2),
            "shape_time": round(shape_time, 2),
            "texture_time": round(texture_time, 2),
        }

        # Check reasonable output size (textured should be 1-100MB)
        if 100 * 1024 < len(textured_bytes) < 100 * 1024 * 1024:
            results["checks"]["reasonable_output_size"] = True
        else:
            results["checks"]["reasonable_output_size"] = False
            results["errors"].append(f"Output size {len(textured_bytes)} is unusual")

        # All checks passed
        if all(results["checks"].values()):
            results["success"] = True

    except Exception as e:
        import traceback

        results["errors"].append(f"{type(e).__name__}: {str(e)[:200]}")
        results["traceback"] = traceback.format_exc()

    _print_results(results)
    return results.get("success", False)


def _print_results(results: dict) -> None:
    """Print test results in a formatted way."""
    print()
    print("=" * 70)
    print("E2E TEST RESULTS")
    print("=" * 70)

    # Print stage results
    for stage_name, stage_data in results.get("stages", {}).items():
        print(f"\n{stage_name.upper()}:")
        for key, value in stage_data.items():
            print(f"  {key}: {value}")

    # Print checks
    print("\nChecks:")
    for check, passed in results.get("checks", {}).items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check}")

    if results.get("errors"):
        print("\nErrors:")
        for error in results["errors"]:
            print(f"  - {error}")

    if results.get("traceback"):
        print("\nTraceback:")
        print(results["traceback"])

    print(f"\n{'=' * 70}")
    if results.get("success"):
        print("E2E PIPELINE TEST PASSED")
        total = results.get("stages", {}).get("total", {})
        print(
            f"Total time: {total.get('time', 'N/A')}s "
            f"(shape: {total.get('shape_time', 'N/A')}s, "
            f"texture: {total.get('texture_time', 'N/A')}s)"
        )
    else:
        print("E2E PIPELINE TEST FAILED")
    print("=" * 70)
