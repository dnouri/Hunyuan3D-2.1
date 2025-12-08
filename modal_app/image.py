"""Modal Container Image Definition for Hunyuan3D-2.1.

This module defines the container image with all dependencies required to run
the Hunyuan3D-2.1 shape and texture generation pipelines on Modal's GPU infrastructure.

Build Strategy:
1. Start from NVIDIA CUDA devel image (need devel for compiling CUDA extensions)
2. Add Python 3.10 (matches original project)
3. Install system dependencies for graphics and mesh processing
4. Install Python dependencies in order of stability (PyTorch first)
5. Clone repository and compile CUDA extensions
6. Download and bake model weights

Usage:
    from modal_app.image import hunyuan_image
"""

import modal

# =============================================================================
# Base Image Configuration
# =============================================================================

# Using CUDA 12.4.1 devel - need devel variant for compiling CUDA extensions
# Ubuntu 22.04 provides stable system libraries
CUDA_BASE = "nvidia/cuda:12.4.1-devel-ubuntu22.04"

# Python version matches original project requirements
PYTHON_VERSION = "3.10"

# CUDA architectures to compile for:
# - 8.0: A100
# - 8.6: A10G (our shape generation GPU)
# - 8.9: L40S (our texture generation GPU), RTX 4090
# - 9.0: H100
CUDA_ARCH_LIST = "8.0;8.6;8.9;9.0"

# =============================================================================
# System Dependencies
# =============================================================================

# Build tools required for compiling Python packages and CUDA extensions
BUILD_PACKAGES = [
    "build-essential",  # GCC, make, etc.
    "ninja-build",  # Fast build system used by PyTorch extensions
    "git",  # Clone repositories
    "wget",  # Download files
    "curl",  # HTTP requests
    "cmake",  # Build system for some packages
]

# Graphics libraries for headless rendering (EGL-based)
GRAPHICS_PACKAGES = [
    "libegl1-mesa-dev",  # EGL development files for headless OpenGL
    "libgl1-mesa-dev",  # OpenGL development files
    "libgles2-mesa-dev",  # OpenGL ES development
    "libglib2.0-0",  # GLib library (required by many packages)
    "libxrender1",  # X Render extension
    "libxi6",  # X Input extension
    "libsm6",  # X Session Management
    "libxext6",  # X extensions
    "libxkbcommon0",  # XKB common library
    "libglvnd0",  # GL Vendor Neutral Dispatch
    "libglx0",  # GLX library
    "xvfb",  # Virtual framebuffer for headless rendering
]

# Mesh processing libraries
MESH_PACKAGES = [
    "libeigen3-dev",  # Linear algebra (required by Open3D, pymeshlab)
    "libcgal-dev",  # Computational geometry algorithms
    "libgmp-dev",  # GNU Multiple Precision (CGAL dependency)
    "libmpfr-dev",  # MPFR library (CGAL dependency)
    "libboost-all-dev",  # Boost libraries (CGAL and others)
]

# Additional system libraries
EXTRA_PACKAGES = [
    "libffi-dev",  # Foreign function interface
    "libssl-dev",  # SSL/TLS development
    "libjpeg-dev",  # JPEG image support
    "libpng-dev",  # PNG image support
    "libtiff-dev",  # TIFF image support
    "ffmpeg",  # Video processing (for some image operations)
    "libsndfile1",  # Audio file support (some ML libs need it)
]

# All system packages combined
SYSTEM_PACKAGES = BUILD_PACKAGES + GRAPHICS_PACKAGES + MESH_PACKAGES + EXTRA_PACKAGES

# =============================================================================
# Environment Variables
# =============================================================================

ENV_VARS = {
    # Use EGL for headless OpenGL (no display required)
    "PYOPENGL_PLATFORM": "egl",
    # CUDA architectures for torch.compile and extension building
    "TORCH_CUDA_ARCH_LIST": CUDA_ARCH_LIST,
    # Disable interactive prompts during apt install
    "DEBIAN_FRONTEND": "noninteractive",
    # Ensure UTF-8 encoding
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    # CUDA paths
    "CUDA_HOME": "/usr/local/cuda",
    "PATH": "/usr/local/cuda/bin:$PATH",
    "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
    # Force g++ compiler (Modal's base uses clang by default)
    "CC": "gcc",
    "CXX": "g++",
    # Hugging Face cache directory (will be populated during build)
    "HF_HOME": "/models",
    # Disable telemetry/analytics
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
}

# =============================================================================
# Python Dependencies
# =============================================================================

# PyTorch index URL for CUDA 12.4
PYTORCH_INDEX = "https://download.pytorch.org/whl/cu124"

# Layer 1: PyTorch and core ML (rarely changes, largest packages)
# These form the foundation and change least frequently
PYTORCH_PACKAGES = [
    "torch==2.5.1",
    "torchvision==0.20.1",
    "torchaudio==2.5.1",
]

# Layer 2: ML framework packages (diffusers, transformers, etc.)
ML_FRAMEWORK_PACKAGES = [
    "transformers==4.46.0",
    "diffusers==0.30.0",
    "accelerate==1.1.1",
    "pytorch-lightning==1.9.5",
    "huggingface-hub==0.30.2",
    "safetensors==0.4.4",
    "hf_transfer",  # For fast model downloads
    "timm",
    "einops==0.8.0",
    "torchdiffeq",
    "torchmetrics==1.6.0",
]

# Layer 3: Scientific computing
SCIENTIFIC_PACKAGES = [
    "numpy==1.24.4",
    "scipy==1.14.1",
    "pandas==2.2.2",
]

# Layer 4: Computer vision and image processing
VISION_PACKAGES = [
    "opencv-python==4.10.0.84",
    "imageio==2.36.0",
    "scikit-image==0.24.0",
    "Pillow",
    "rembg==2.0.65",
    # Note: realesrgan and basicsr have complex deps, install separately
]

# Layer 5: 3D mesh processing
MESH_PROCESSING_PACKAGES = [
    "trimesh==4.4.7",
    "pymeshlab==2022.2.post3",
    "pygltflib==1.16.3",
    "xatlas==0.0.9",
    "open3d==0.18.0",
    "pythreejs",
]

# Layer 6: Configuration and utilities
CONFIG_PACKAGES = [
    "omegaconf==2.3.0",
    "pyyaml==6.0.2",
    "configargparse==1.7",
    "tqdm==4.66.5",
    "psutil==6.0.0",
    "pydantic>=2.0",
]

# Layer 7: Web framework (for API)
WEB_PACKAGES = [
    "fastapi==0.115.12",
    "uvicorn==0.34.3",
]

# Layer 8: AWS SDK (for S3 uploads)
AWS_PACKAGES = [
    "boto3",
]

# Layer 9: Build tools and additional packages
BUILD_PYTHON_PACKAGES = [
    "ninja==1.11.1.1",
    "pybind11==2.13.4",
    "wheel",  # Required for building CUDA extensions
    "setuptools",  # Required for building CUDA extensions
]

# Layer 10: GPU computing and additional ML packages
GPU_PACKAGES = [
    "cupy-cuda12x==13.4.1",
    "onnxruntime-gpu==1.16.3",  # GPU version for CUDA
]

# Packages that need special handling (complex dependencies)
# These are installed separately to avoid conflicts
SPECIAL_PACKAGES = [
    "realesrgan==0.3.0",
    "basicsr==1.4.2",
    "tb_nightly",  # TensorBoard nightly for basicsr
]

# =============================================================================
# Base Image + System Dependencies
# =============================================================================

# Start building the base image with system dependencies.
# IMPORTANT: Modal's builder has internal steps that require Python.
# We MUST install Python in the FIRST run_commands step, before using apt_install.

base_image = (
    modal.Image.from_registry(CUDA_BASE, add_python=PYTHON_VERSION)
    # Set environment variables
    .env(ENV_VARS)
    # Install all system dependencies in one layer
    .apt_install(*SYSTEM_PACKAGES)
)

# =============================================================================
# Python Dependencies Image
# =============================================================================

# Enable hf_transfer for fast HuggingFace downloads
python_env_vars = {
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
}

# Combine all stable packages into fewer layers for faster builds
# Layer 1: Core packages (PyTorch + ML frameworks + utilities)
CORE_PACKAGES = (
    PYTORCH_PACKAGES
    + ML_FRAMEWORK_PACKAGES
    + SCIENTIFIC_PACKAGES
    + CONFIG_PACKAGES
    + WEB_PACKAGES
    + AWS_PACKAGES
    + BUILD_PYTHON_PACKAGES
)

# Layer 2: Vision + 3D packages (separate due to complexity)
VISION_3D_PACKAGES = VISION_PACKAGES + MESH_PROCESSING_PACKAGES

# Build Python dependencies image
python_image = (
    base_image.env(python_env_vars)
    # Layer 1: PyTorch with CUDA + core ML frameworks
    .pip_install(
        *CORE_PACKAGES,
        extra_index_url=PYTORCH_INDEX,
    )
    # Layer 2: Vision and 3D packages
    .pip_install(*VISION_3D_PACKAGES)
    # Layer 3: GPU-specific packages (cupy, onnxruntime-gpu)
    .pip_install(*GPU_PACKAGES)
    # Layer 4: Special packages with complex dependencies
    .pip_install(*SPECIAL_PACKAGES)
    # Layer 5: Re-enforce numpy<2 constraint
    # onnxruntime-gpu and other packages may upgrade numpy to 2.x as a transitive
    # dependency. onnxruntime<1.19 was compiled against numpy 1.x and crashes with
    # numpy 2.x. This layer ensures numpy stays at 1.x regardless of what earlier
    # layers installed. See: https://github.com/microsoft/onnxruntime/issues/21063
    .pip_install("numpy<2")
    # Layer 6: Fix basicsr/torchvision compatibility
    # basicsr==1.4.2 imports from torchvision.transforms.functional_tensor which
    # was removed in torchvision 0.16+. Create a shim module that redirects to
    # the new location (torchvision.transforms.functional).
    .run_commands(
        'python -c "'
        "import os; "
        "path = '/usr/local/lib/python3.10/site-packages/torchvision/transforms/functional_tensor.py'; "
        "os.makedirs(os.path.dirname(path), exist_ok=True); "
        "open(path, 'w').write('from torchvision.transforms.functional import rgb_to_grayscale\\n')"
        '"'
    )
)

# =============================================================================
# Extensions Image (Repository + CUDA Compilation)
# =============================================================================

# GitHub repository URL
HUNYUAN3D_REPO = "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git"
APP_DIR = "/app"

# RealESRGAN checkpoint URL
REALESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

# Clone repository and compile CUDA extensions
extensions_image = (
    python_image
    # Clone repository to /app
    .run_commands(
        f"git clone --depth 1 {HUNYUAN3D_REPO} {APP_DIR}",
    )
    # Compile custom_rasterizer (CUDA extension)
    # Uses environment variables TORCH_CUDA_ARCH_LIST set in base image
    # --no-build-isolation required because setup.py imports torch
    .run_commands(
        f"cd {APP_DIR}/hy3dpaint/custom_rasterizer && pip install --no-build-isolation -e .",
    )
    # Compile mesh_inpaint_processor (pybind11 C++ extension)
    .run_commands(
        f"cd {APP_DIR}/hy3dpaint/DifferentiableRenderer && "
        "c++ -O3 -Wall -shared -std=c++11 -fPIC "
        "$(python -m pybind11 --includes) "
        "mesh_inpaint_processor.cpp "
        "-o mesh_inpaint_processor$(python3-config --extension-suffix)",
    )
    # Download RealESRGAN checkpoint
    .run_commands(
        f"mkdir -p {APP_DIR}/hy3dpaint/ckpt && "
        f"wget -q {REALESRGAN_URL} -O {APP_DIR}/hy3dpaint/ckpt/RealESRGAN_x4plus.pth",
    )
    # Set working directory
    .workdir(APP_DIR)
)

# =============================================================================
# Final Image with Baked Model Weights
# =============================================================================

# HuggingFace model repositories to download
# These are baked into the image for fastest cold starts
HUNYUAN3D_MODEL = "tencent/Hunyuan3D-2.1"
DINOV2_MODEL = "facebook/dinov2-giant"

# Model weights directory (matches HF_HOME env var)
MODELS_DIR = "/models"

# HuggingFace repo IDs for model downloads
# These are passed to snapshot_download() and from_pretrained()
HUNYUAN3D_REPO_ID = "tencent/Hunyuan3D-2.1"
DINOV2_REPO_ID = "facebook/dinov2-giant"

# Runtime path constants - used by services and tests
# These define where models and code live inside the container
# When using --local-dir, models are downloaded to flat directory structure
HUNYUAN3D_MODEL_DIR = f"{MODELS_DIR}/hunyuan3d"
HUNYUAN3D_DIT_DIR = f"{HUNYUAN3D_MODEL_DIR}/hunyuan3d-dit-v2-1"
HUNYUAN3D_VAE_DIR = f"{HUNYUAN3D_MODEL_DIR}/hunyuan3d-vae-v2-1"
DINOV2_MODEL_DIR = f"{MODELS_DIR}/dinov2"

# Package paths (hy3dshape has nested structure: /app/hy3dshape/hy3dshape/)
HY3DSHAPE_PACKAGE_PATH = f"{APP_DIR}/hy3dshape"
DIFFERENTIABLE_RENDERER_PATH = f"{APP_DIR}/hy3dpaint/DifferentiableRenderer"
REALESRGAN_CKPT_PATH = f"{APP_DIR}/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"

# Download model weights using huggingface-cli with hf_transfer for speed
# Models are stored in HuggingFace cache format at $HF_HOME/hub/
# This allows snapshot_download() and from_pretrained() to find them
weights_image = (
    extensions_image
    # Download Hunyuan3D-2.1 model (shape and texture pipelines)
    # This includes:
    # - hunyuan3d-dit-v2-1/ (shape generation ~4GB)
    # - hunyuan3d-paintpbr-v2-1/ (texture generation ~5GB)
    # Using HF cache format so snapshot_download() finds them at runtime
    .run_commands(
        f"huggingface-cli download {HUNYUAN3D_MODEL}",
    )
    # Download DINOv2-giant model (used by texture pipeline for image encoding)
    # Size: ~4.4GB
    .run_commands(
        f"huggingface-cli download {DINOV2_MODEL}",
    )
)

# =============================================================================
# Full Image Definition
# =============================================================================

# Complete image with all dependencies, extensions, and baked weights
# Phases: base_image (1.1) + python_image (1.2) + extensions_image (1.3) + weights_image (1.4)
hunyuan_image = weights_image

# Export components for testing and debugging
__all__ = [
    # Images
    "hunyuan_image",
    "base_image",
    "python_image",
    "extensions_image",
    "weights_image",
    # Base configuration
    "CUDA_BASE",
    "PYTHON_VERSION",
    "CUDA_ARCH_LIST",
    "PYTORCH_INDEX",
    "ENV_VARS",
    # Repository and build paths
    "HUNYUAN3D_REPO",
    "APP_DIR",
    "REALESRGAN_URL",
    "HUNYUAN3D_MODEL",
    "DINOV2_MODEL",
    "MODELS_DIR",
    # HuggingFace repo IDs (for from_pretrained and snapshot_download)
    "HUNYUAN3D_REPO_ID",
    "DINOV2_REPO_ID",
    # Runtime path constants (for services and tests)
    # Note: With HF cache format, prefer using repo IDs instead of local paths
    "HUNYUAN3D_MODEL_DIR",
    "HUNYUAN3D_DIT_DIR",
    "HUNYUAN3D_VAE_DIR",
    "DINOV2_MODEL_DIR",
    "HY3DSHAPE_PACKAGE_PATH",
    "DIFFERENTIABLE_RENDERER_PATH",
    "REALESRGAN_CKPT_PATH",
    # Package lists
    "SYSTEM_PACKAGES",
    "PYTORCH_PACKAGES",
    "ML_FRAMEWORK_PACKAGES",
    "SCIENTIFIC_PACKAGES",
    "VISION_PACKAGES",
    "MESH_PROCESSING_PACKAGES",
    "CONFIG_PACKAGES",
    "WEB_PACKAGES",
    "AWS_PACKAGES",
    "BUILD_PYTHON_PACKAGES",
    "GPU_PACKAGES",
    "SPECIAL_PACKAGES",
]
