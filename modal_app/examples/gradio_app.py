#!/usr/bin/env python3
"""Gradio Frontend for Hunyuan3D-2.1 Modal API.

A web interface for generating 3D meshes using the deployed Modal API.
Does NOT require a local GPU - all computation runs on Modal's cloud GPUs.

Features:
- Image upload with preview
- Generation parameters (steps, guidance_scale, seed, resolution)
- Shape-only vs full texture toggle
- Real-time progress display
- 3D model preview and download

Usage:
    # Set environment variables
    export HUNYUAN3D_API_URL="https://your-app.modal.run"
    export HUNYUAN3D_API_KEY="sk_live_xxx"

    # Run the app
    python modal_app/examples/gradio_app.py

Requirements:
    pip install gradio httpx
"""

from __future__ import annotations

import os
import tempfile

import gradio as gr
import httpx

from modal_app.api.sse import parse_sse_line

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_API_URL = os.environ.get("HUNYUAN3D_API_URL", "https://your-app.modal.run")
DEFAULT_API_KEY = os.environ.get("HUNYUAN3D_API_KEY", "")

# Generation parameter defaults
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 5.0
DEFAULT_SEED = 42
DEFAULT_RESOLUTION = 256

# Resolution options for the UI
RESOLUTION_OPTIONS = [196, 256, 384]


def generate_3d_mesh(
    image_path: str,
    api_url: str,
    api_key: str,
    generate_texture: bool = True,
    seed: int | None = None,
    steps: int = DEFAULT_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    octree_resolution: int = DEFAULT_RESOLUTION,
    progress_callback=None,
) -> tuple[str | None, str | None]:
    """Generate a 3D mesh from an image via the Modal API.

    Args:
        image_path: Path to input image file
        api_url: Base URL of the Hunyuan3D API
        api_key: Valid API key
        generate_texture: Whether to generate PBR textures
        seed: Optional random seed for reproducibility
        steps: Number of inference steps
        guidance_scale: Classifier-free guidance scale
        octree_resolution: Mesh octree resolution
        progress_callback: Optional callback for progress updates (stage, percent)

    Returns:
        Tuple of (local_glb_path, error_message)
    """
    if not api_url:
        return None, "API URL is required"
    if not api_key:
        return None, "API key is required"
    if not image_path:
        return None, "Image is required"

    # Prepare request
    with open(image_path, "rb") as f:
        files = {"image": f}
        data = {
            "generate_texture": str(generate_texture).lower(),
            "steps": str(steps),
            "guidance_scale": str(guidance_scale),
            "octree_resolution": str(octree_resolution),
        }
        if seed is not None:
            data["seed"] = str(seed)

        headers = {"X-API-Key": api_key}

        try:
            with httpx.Client(timeout=600) as client:
                with client.stream(
                    "POST",
                    f"{api_url}/generate/stream",
                    headers=headers,
                    files=files,
                    data=data,
                ) as response:
                    if response.status_code == 401:
                        return None, "Invalid API key"
                    if response.status_code != 200:
                        return None, f"API error: HTTP {response.status_code}"

                    current_event = None
                    download_url = None

                    for line in response.iter_lines():
                        line_type, value = parse_sse_line(line)

                        if line_type == "event":
                            current_event = value
                        elif line_type == "data" and current_event:
                            if current_event == "progress":
                                stage = value.get("stage", "unknown")
                                percent = value.get("percent", 0)
                                if progress_callback:
                                    progress_callback(stage, percent)

                            elif current_event == "completed":
                                download_url = value.get("download_url")

                            elif current_event == "error":
                                return None, f"Generation error: {value.get('message')}"

                            current_event = None

            if not download_url:
                return None, "No download URL received"

            # Download the result to a temp file
            with httpx.Client(timeout=120) as client:
                response = client.get(download_url)
                response.raise_for_status()

                # Save to temp file
                temp_file = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
                temp_file.write(response.content)
                temp_file.close()

                return temp_file.name, None

        except httpx.TimeoutException:
            return None, "Request timed out"
        except httpx.HTTPError as e:
            return None, f"HTTP error: {e}"
        except Exception as e:
            return None, f"Unexpected error: {e}"


# =============================================================================
# Gradio Interface
# =============================================================================


def generate_model(
    image,
    api_url: str,
    api_key: str,
    generate_texture: bool,
    seed: int,
    randomize_seed: bool,
    steps: int,
    guidance_scale: float,
    octree_resolution: int,
    progress=gr.Progress(track_tqdm=True),
):
    """Gradio handler for model generation."""
    import random

    # Handle seed
    if randomize_seed:
        seed = random.randint(0, 2**31 - 1)

    # Progress callback
    def update_progress(stage: str, percent: int):
        stage_display = stage.replace("_", " ").title()
        progress(percent / 100, desc=f"{stage_display}")

    # Generate
    glb_path, error = generate_3d_mesh(
        image_path=image,
        api_url=api_url,
        api_key=api_key,
        generate_texture=generate_texture,
        seed=seed,
        steps=steps,
        guidance_scale=guidance_scale,
        octree_resolution=octree_resolution,
        progress_callback=update_progress,
    )

    if error:
        raise gr.Error(error)

    return glb_path, seed


def create_interface():
    """Create the Gradio interface."""
    with gr.Blocks(
        title="Hunyuan3D-2.1 (Modal API)",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # Hunyuan3D-2.1 (Modal API)

            Generate 3D meshes from images using Tencent's Hunyuan3D-2.1 model.
            All computation runs on Modal's cloud GPUs - no local GPU required.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### Input")
                image_input = gr.Image(
                    label="Upload Image",
                    type="filepath",
                    height=256,
                )

                # API Configuration
                gr.Markdown("### API Configuration")
                api_url = gr.Textbox(
                    label="API URL",
                    value=DEFAULT_API_URL,
                    placeholder="https://your-app.modal.run",
                )
                api_key = gr.Textbox(
                    label="API Key",
                    value=DEFAULT_API_KEY,
                    placeholder="sk_live_xxx",
                    type="password",
                )

                # Generation parameters
                gr.Markdown("### Generation Parameters")
                generate_texture = gr.Checkbox(
                    label="Generate Textures (PBR)",
                    value=True,
                    info="Uncheck for shape-only (faster)",
                )

                with gr.Row():
                    seed = gr.Number(
                        label="Seed",
                        value=DEFAULT_SEED,
                        precision=0,
                    )
                    randomize_seed = gr.Checkbox(
                        label="Randomize",
                        value=False,
                    )

                steps = gr.Slider(
                    label="Inference Steps",
                    minimum=10,
                    maximum=100,
                    step=5,
                    value=DEFAULT_STEPS,
                    info="Higher = better quality but slower",
                )

                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=1.0,
                    maximum=15.0,
                    step=0.5,
                    value=DEFAULT_GUIDANCE_SCALE,
                    info="Classifier-free guidance strength",
                )

                octree_resolution = gr.Radio(
                    label="Mesh Resolution",
                    choices=RESOLUTION_OPTIONS,
                    value=DEFAULT_RESOLUTION,
                    info="Higher = more detail but slower",
                )

                generate_btn = gr.Button(
                    "Generate 3D Model",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=2):
                # Output section
                gr.Markdown("### 3D Preview")
                model_output = gr.Model3D(
                    label="Generated Model",
                    height=500,
                    clear_color=[0.9, 0.9, 0.9, 1.0],
                )

                with gr.Row():
                    seed_output = gr.Number(
                        label="Seed Used",
                        interactive=False,
                    )

        # Wire up the button
        generate_btn.click(
            fn=generate_model,
            inputs=[
                image_input,
                api_url,
                api_key,
                generate_texture,
                seed,
                randomize_seed,
                steps,
                guidance_scale,
                octree_resolution,
            ],
            outputs=[model_output, seed_output],
        )

        # Example images
        gr.Markdown(
            """
            ### Notes

            - **Shape-only mode**: Faster (~30s) but produces untextured white mesh
            - **With textures**: Full PBR materials (~90s total)
            - **Resolution**: Higher values produce more detailed meshes but take longer

            ### Unsupported Features

            - **Text-to-image**: Use an external image generator first
            - **Multi-view mode**: Currently only single image input is supported
            """
        )

    return demo


# =============================================================================
# Main
# =============================================================================


def main():
    """Run the Gradio app."""
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
