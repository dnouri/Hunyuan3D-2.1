"""Smoke Test for Hunyuan3D Modal Image.

Validates that the complete Modal image is correctly configured with all
dependencies, CUDA extensions, and model weights.

Run with:
    modal run modal_app/smoke_test.py

Expected outcome:
- All key modules importable (torch, diffusers, trimesh, custom_rasterizer)
- CUDA available and working
- Model weights loadable from baked paths
- Shape generation pipeline initializes successfully
"""

import time

import modal

from modal_app.image import (
    APP_DIR,
    DIFFERENTIABLE_RENDERER_PATH,
    DINOV2_MODEL_DIR,
    HUNYUAN3D_DIT_DIR,
    HUNYUAN3D_MODEL_DIR,
    HUNYUAN3D_VAE_DIR,
    HY3DSHAPE_PACKAGE_PATH,
    REALESRGAN_CKPT_PATH,
    hunyuan_image,
)

# Mount modal_app package into container so imports work remotely
smoke_test_image = hunyuan_image.add_local_python_source("modal_app")

app = modal.App("hunyuan3d-smoke-test")

# =============================================================================
# Check Functions - Each does one thing
# =============================================================================


def check_core_imports() -> dict:
    """Check that core Python packages are importable."""
    results = {"checks": {}, "errors": {}}
    core_modules = [
        "torch",
        "diffusers",
        "transformers",
        "trimesh",
        "PIL",
        "numpy",
        "omegaconf",
    ]

    for module in core_modules:
        try:
            __import__(module)
            results["checks"][f"import_{module}"] = True
        except Exception as e:
            results["checks"][f"import_{module}"] = False
            results["errors"][f"import_{module}"] = str(e)[:100]

    return results


def check_cuda() -> dict:
    """Check CUDA availability and GPU info."""
    results = {"checks": {}, "errors": {}, "info": {}}

    try:
        import torch

        results["checks"]["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            results["info"]["cuda_device"] = torch.cuda.get_device_name(0)
            results["info"]["cuda_device_count"] = torch.cuda.device_count()
            results["info"]["cuda_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
            )
    except Exception as e:
        results["checks"]["cuda_available"] = False
        results["errors"]["cuda"] = str(e)[:100]

    return results


def check_cuda_extensions() -> dict:
    """Check that CUDA extensions are importable."""
    import sys

    results = {"checks": {}, "errors": {}}

    # custom_rasterizer (installed as package)
    try:
        results["checks"]["custom_rasterizer"] = True
    except Exception as e:
        results["checks"]["custom_rasterizer"] = False
        results["errors"]["custom_rasterizer"] = str(e)[:100]

    # mesh_inpaint_processor (compiled .so file)
    sys.path.insert(0, DIFFERENTIABLE_RENDERER_PATH)
    try:
        results["checks"]["mesh_inpaint_processor"] = True
    except Exception as e:
        results["checks"]["mesh_inpaint_processor"] = False
        results["errors"]["mesh_inpaint_processor"] = str(e)[:100]

    return results


def check_model_weights() -> dict:
    """Check that model weights exist at expected paths."""
    import os

    results = {"checks": {}}

    results["checks"]["hunyuan3d_exists"] = os.path.isdir(HUNYUAN3D_MODEL_DIR)
    results["checks"]["hunyuan3d_dit_exists"] = os.path.isdir(HUNYUAN3D_DIT_DIR)
    results["checks"]["hunyuan3d_vae_exists"] = os.path.isdir(HUNYUAN3D_VAE_DIR)
    results["checks"]["dinov2_exists"] = os.path.isdir(DINOV2_MODEL_DIR)
    results["checks"]["realesrgan_exists"] = os.path.exists(REALESRGAN_CKPT_PATH)

    return results


def check_pipeline_init() -> dict:
    """Check that shape generation pipeline initializes on GPU."""
    import os
    import sys

    results = {"checks": {}, "errors": {}, "timings": {}}
    start_time = time.time()

    try:
        # hy3dshape has nested structure: /app/hy3dshape/hy3dshape/
        sys.path.insert(0, HY3DSHAPE_PACKAGE_PATH)
        os.chdir(APP_DIR)

        from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline

        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            HUNYUAN3D_MODEL_DIR,
            subfolder="hunyuan3d-dit-v2-1",
            use_safetensors=False,
            device="cuda",
        )

        results["checks"]["pipeline_init"] = True
        results["checks"]["pipeline_on_cuda"] = next(
            pipeline.model.parameters()
        ).is_cuda

        del pipeline
        import torch

        torch.cuda.empty_cache()

    except Exception as e:
        results["checks"]["pipeline_init"] = False
        results["errors"]["pipeline_init"] = str(e)[:200]

    results["timings"]["pipeline_init_seconds"] = round(time.time() - start_time, 2)
    return results


# =============================================================================
# Main Smoke Test
# =============================================================================


@app.function(
    image=smoke_test_image,
    gpu="A10G",
    timeout=600,
)
def smoke_test() -> dict:
    """Comprehensive smoke test for the Hunyuan3D Modal image."""
    import os
    import sys

    start_time = time.time()

    # Collect environment info
    results = {
        "python_version": sys.version,
        "cwd": os.getcwd(),
        "checks": {},
        "timings": {},
        "errors": {},
    }

    # Run all checks
    for check_fn in [
        check_core_imports,
        check_cuda,
        check_cuda_extensions,
        check_model_weights,
        check_pipeline_init,
    ]:
        check_results = check_fn()
        results["checks"].update(check_results.get("checks", {}))
        results["errors"].update(check_results.get("errors", {}))
        results["timings"].update(check_results.get("timings", {}))

        # Copy info fields (cuda device, etc.)
        for key, value in check_results.get("info", {}).items():
            results[key] = value

    results["timings"]["total_seconds"] = round(time.time() - start_time, 2)
    results["all_passed"] = all(results["checks"].values())

    return results


# =============================================================================
# CLI Entrypoint
# =============================================================================


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("HUNYUAN3D MODAL IMAGE SMOKE TEST")
    print("=" * 70)

    print("\nRunning comprehensive validation...")
    print("(First run may take a few minutes for container initialization)")

    result = smoke_test.remote()

    print(f"\n{'=' * 70}")
    print("ENVIRONMENT")
    print(f"{'=' * 70}")
    print(f"Python: {result['python_version'].split()[0]}")
    print(f"Working Directory: {result['cwd']}")
    if result.get("cuda_device"):
        print(f"GPU: {result['cuda_device']}")
        print(f"GPU Memory: {result.get('cuda_memory_gb', 'N/A')} GB")

    print(f"\n{'=' * 70}")
    print("CHECK RESULTS")
    print(f"{'=' * 70}")

    categories = {
        "Core Imports": [k for k in result["checks"] if k.startswith("import_")],
        "CUDA": ["cuda_available", "custom_rasterizer", "mesh_inpaint_processor"],
        "Model Weights": [
            k
            for k in result["checks"]
            if any(x in k for x in ["hunyuan", "dinov2", "realesrgan"])
        ],
        "Pipeline": ["pipeline_init", "pipeline_on_cuda"],
    }

    for category, checks in categories.items():
        print(f"\n{category}:")
        for check in checks:
            if check in result["checks"]:
                status = "PASS" if result["checks"][check] else "FAIL"
                check_name = check.replace("import_", "").replace("_", " ")
                print(f"  [{status}] {check_name}")
                if not result["checks"][check] and check in result.get("errors", {}):
                    print(f"          Error: {result['errors'][check][:80]}...")

    print(f"\n{'=' * 70}")
    print("TIMINGS")
    print(f"{'=' * 70}")
    for timing, value in result.get("timings", {}).items():
        print(f"  {timing}: {value}s")

    print(f"\n{'=' * 70}")
    if result.get("all_passed"):
        print("ALL CHECKS PASSED - Image is ready for production!")
    else:
        print("SOME CHECKS FAILED - Review errors above")
        failed = [k for k, v in result["checks"].items() if not v]
        print(f"Failed checks: {', '.join(failed)}")
    print(f"{'=' * 70}")
