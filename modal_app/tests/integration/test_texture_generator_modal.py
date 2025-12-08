"""Integration test for TextureGenerator service on Modal.

This test runs the TextureGenerator with real GPU to verify:
- Pipeline initialization with memory snapshots
- Texture generation from mesh and reference image
- Output is valid textured GLB mesh

Run with:
    modal run tests/integration/test_texture_generator_modal.py

Expected:
- First run: ~90-120s (cold start + inference)
- Subsequent runs: ~60-75s (warm start + inference)

Note:
    This test uses a fixture mesh created with trimesh instead of calling
    ShapeGenerator, to avoid cross-app Modal function calls. For full
    end-to-end testing, see test_pipeline_e2e.py.
"""

from pathlib import Path

from modal_app.services.texture_generator import (
    TextureGenerator,
    app,
    texture_generator_image,
)


def create_fixture_mesh() -> bytes:
    """Create a simple box mesh using trimesh for testing.

    Returns:
        GLB mesh as bytes
    """
    import io

    import trimesh

    # Create a simple UV-sphere mesh (better for texture testing than a box)
    # The sphere has proper UV coordinates for texture mapping
    mesh = trimesh.creation.uv_sphere(radius=1.0, count=[32, 32])

    # Export to GLB
    buffer = io.BytesIO()
    mesh.export(buffer, file_type="glb")
    buffer.seek(0)
    return buffer.read()


@app.function(image=texture_generator_image)
def run_integration_test() -> dict:
    """Run texture generation integration test.

    This test uses a fixture mesh (UV sphere) and the demo reference image
    to test the texture generation pipeline in isolation.

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
        "mesh_size": 0,
        "image_size": 0,
        "output_size": 0,
        "inference_time": 0.0,
        "errors": [],
        "checks": {},
    }

    try:
        # Load the reference image from container
        image_path = Path("/app/assets/demo.png")
        if not image_path.exists():
            results["errors"].append(f"Test image not found: {image_path}")
            return results

        image_bytes = image_path.read_bytes()
        results["image_size"] = len(image_bytes)
        results["checks"]["image_loaded"] = True

        # Create fixture mesh using trimesh (available in container)
        print("Creating fixture mesh (UV sphere)...")
        mesh_bytes = create_fixture_mesh()
        results["mesh_size"] = len(mesh_bytes)
        results["checks"]["fixture_mesh_created"] = True

        # Validate mesh is GLB before using it
        if mesh_bytes[:4] != b"glTF":
            results["errors"].append("Fixture mesh is not a valid GLB file")
            results["checks"]["valid_input_mesh"] = False
            return results
        results["checks"]["valid_input_mesh"] = True

        # Run the texture generator
        print("Applying texture to fixture mesh...")
        texture_start = time.time()
        output_bytes = TextureGenerator().generate.remote(
            mesh_bytes,
            image_bytes,
            seed=42,
        )
        texture_time = time.time() - texture_start

        results["inference_time"] = round(texture_time, 2)
        results["output_size"] = len(output_bytes)
        results["checks"]["texture_applied"] = True

        # Validate output is GLB
        if output_bytes[:4] == b"glTF":
            results["checks"]["valid_glb_header"] = True
        else:
            results["checks"]["valid_glb_header"] = False
            results["errors"].append("Output does not have valid GLB header")

        # Check output size is reasonable
        # Textured GLB should be larger than untextured (due to embedded textures)
        # Typically 1MB - 50MB
        if 100 * 1024 < len(output_bytes) < 100 * 1024 * 1024:
            results["checks"]["reasonable_size"] = True
        else:
            results["checks"]["reasonable_size"] = False
            results["errors"].append(f"Output size {len(output_bytes)} is unusual")

        # Check textured mesh is larger than untextured (has texture data)
        if len(output_bytes) > len(mesh_bytes):
            results["checks"]["has_texture_data"] = True
        else:
            results["checks"]["has_texture_data"] = False
            results["errors"].append("Textured mesh not larger than untextured")

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
    print("TEXTURE GENERATOR INTEGRATION TEST")
    print("=" * 70)
    print("\nRunning texture generation on Modal GPU (L40S)...")
    print("(First run may take 3-5 minutes for container + model initialization)")
    print("(Uses fixture mesh to test texture pipeline in isolation)")
    print()

    result = run_integration_test.remote()

    print(f"{'=' * 70}")
    print("TEST RESULTS")
    print(f"{'=' * 70}")

    print(f"\nInput image size: {result.get('image_size', 'N/A')} bytes")
    print(f"Fixture mesh size: {result.get('mesh_size', 'N/A')} bytes")
    print(f"Textured mesh size: {result.get('output_size', 'N/A')} bytes")
    print(f"Texture generation time: {result.get('inference_time', 'N/A')}s")

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
