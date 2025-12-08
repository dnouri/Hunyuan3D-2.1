# Hunyuan3D-2.1 Modal API Examples

Example clients and utilities for the Hunyuan3D-2.1 Modal API. All examples run
locally without a GPU - computation happens on Modal's cloud infrastructure.

## Quick Start

```bash
# Set your API credentials
export HUNYUAN3D_API_URL="https://your-app.modal.run"
export HUNYUAN3D_API_KEY="sk_live_xxx"

# Install dependencies
pip install httpx gradio trimesh

# Run any example
python modal_app/examples/python_client.py --image photo.png --api-key $HUNYUAN3D_API_KEY
```

## Examples

### Python Client (`python_client.py`)

Command-line client with SSE progress streaming and automatic download.

```bash
# Basic usage
python modal_app/examples/python_client.py \
    --image input.png \
    --api-key sk_live_xxx \
    --output result.glb

# Shape-only mode (faster, no textures)
python modal_app/examples/python_client.py \
    --image input.png \
    --api-key sk_live_xxx \
    --no-texture

# With reproducible seed
python modal_app/examples/python_client.py \
    --image input.png \
    --api-key sk_live_xxx \
    --seed 42
```

**Requirements:** `pip install httpx`

### Browser Client (`browser_client.html`)

Single-file HTML/JavaScript client. Open in any browser - no build step required.

Features:
- Image upload with preview
- Real-time progress bar
- SSE event logging
- Direct download link

Usage: Open `browser_client.html` in your browser, enter API credentials, and upload an image.

### Gradio Web App (`gradio_app.py`)

Full-featured web interface with 3D model preview.

```bash
# Start the Gradio server
python modal_app/examples/gradio_app.py

# Opens at http://localhost:7860
```

Features:
- Image upload with preview
- All generation parameters (steps, guidance_scale, resolution)
- Real-time progress display
- Interactive 3D model viewer
- Seed randomization option

**Requirements:** `pip install gradio httpx`

### Mesh Utilities (`mesh_utils.py`)

Local post-processing for GLB meshes. Useful for format conversion and
optimization after downloading from the API.

```bash
# Convert GLB to OBJ
python modal_app/examples/mesh_utils.py convert model.glb model.obj

# Reduce mesh to 10k faces
python modal_app/examples/mesh_utils.py reduce model.glb simplified.glb --faces 10000

# Get mesh information
python modal_app/examples/mesh_utils.py info model.glb
```

Programmatic usage:

```python
from mesh_utils import convert_format, reduce_faces, get_mesh_info

# Convert format
obj_bytes = convert_format(glb_bytes, "obj")

# Simplify mesh
simplified = reduce_faces(glb_bytes, target_faces=10000)

# Inspect mesh
info = get_mesh_info(glb_bytes)
print(f"Vertices: {info['vertex_count']}, Faces: {info['face_count']}")
```

**Requirements:** `pip install trimesh numpy`

## Generation Parameters

All clients support these parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `generate_texture` | `true` | Generate PBR textures (set to `false` for shape-only) |
| `seed` | random | Random seed for reproducibility |
| `steps` | `30` | Inference steps (10-100, higher = better quality) |
| `guidance_scale` | `5.0` | Classifier-free guidance (1.0-15.0) |
| `octree_resolution` | `256` | Mesh detail (196, 256, or 384) |

**Performance tips:**
- Shape-only mode (`generate_texture=false`) takes ~30s vs ~90s with textures
- Lower `steps` (20-25) is faster with minimal quality loss
- Resolution 196 is fastest, 384 is highest detail

## Unsupported Features

The Modal API focuses on image-to-3D generation. The following features from the
original Hunyuan3D-2.1 are **not available** via the API:

### Text-to-3D Generation
The API does not support text prompts directly. For text-to-3D:
1. Use an external image generator (DALL-E, Midjourney, Stable Diffusion)
2. Pass the generated image to this API

### Multi-View Input
The API accepts a single image. Multi-view reconstruction is not supported.

### Lite Model Variants
Only the full Hunyuan3D-2.1 model is deployed. Lite variants are not available.

### Custom Model Weights
The API uses the official pretrained weights. Custom fine-tuned models cannot
be deployed through this interface.

## API Response Format

All endpoints use Server-Sent Events (SSE) for streaming progress:

```
event: started
data: {"job_id": "uuid", "timestamp": "..."}

event: progress
data: {"stage": "shape", "percent": 25}

event: progress
data: {"stage": "texture", "percent": 75}

event: completed
data: {"job_id": "uuid", "download_url": "https://..."}
```

Error responses:

```
event: error
data: {"stage": "shape", "message": "...", "retriable": false}
```

## Troubleshooting

**401 Unauthorized**: Check that your API key is correct and has the `sk_live_` prefix.

**Timeout errors**: Generation can take 30-90 seconds. Ensure your client has a
timeout of at least 300 seconds for the SSE stream.

**Empty/black mesh**: Input images work best when:
- Object is centered with clear silhouette
- Background is simple or transparent
- Good lighting without harsh shadows

**Download failed**: Presigned URLs expire after 1 hour. If expired, re-run generation.
