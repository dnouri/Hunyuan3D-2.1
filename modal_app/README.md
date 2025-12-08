# Hunyuan3D-2.1 Modal Integration

A serverless deployment of Hunyuan3D-2.1 on Modal's GPU infrastructure.

## Why This Exists

Running Hunyuan3D locally requires a 24GB+ GPU, complex CUDA dependencies, and
careful environment management. Not everyone has that. This integration solves
that problem by running the heavy computation on Modal's cloud GPUs, exposing
a simple REST API that any client can call.

The client sends an image. The API returns a 3D mesh. No GPU required locally.

## What It Does

A serverless API that:
- Accepts an input image via HTTP POST
- Generates a 3D mesh (shape generation) on cloud GPU
- Optionally applies PBR textures (texture generation) on cloud GPU
- Streams progress updates via Server-Sent Events
- Returns a presigned S3 URL for the resulting GLB file

## Architecture

```
Client                    Modal API (FastAPI)           Modal GPU Services
  |                              |                              |
  |--POST /generate/stream------>|                              |
  |   + image + API key          |                              |
  |                              |                              |
  |<-SSE: started----------------|                              |
  |                              |---shape_svc.generate()------>| A10G (24GB)
  |<-SSE: progress 10%-----------|                              |
  |<-SSE: progress 50%-----------|<---------mesh_bytes----------|
  |                              |                              |
  |                              |---texture_svc.generate()---->| L40S (48GB)
  |<-SSE: progress 55%-----------|                              |
  |<-SSE: progress 95%-----------|<--------result_bytes---------|
  |                              |                              |
  |                              |---upload to S3-------------->| (AWS)
  |<-SSE: completed + URL--------|                              |
  |                              |                              |
  |---GET presigned URL---------------------------------------->| S3
  |<----------GLB file------------------------------------------|
```

Shape and texture generation run on separate GPU types. This is intentional.
Shape generation needs 10GB VRAM; texture generation needs 21GB. Splitting
them means we can use cheaper GPUs for shape (A10G) and reserve the expensive
ones (L40S) for texture. Pay for what you need.

## Design Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Container Image | Modal native build | Reproducible, version-controlled |
| Model Weights | Baked into image | Fastest cold starts, stable models |
| Cold Start | CPU memory snapshots | Safe with automatic fallback |
| GPU Split | Shape: A10G, Texture: L40S | Cost-optimized ($2.80/hr combined) |
| API Contract | SSE streaming | Real-time progress, no HTTP timeouts |
| Storage | AWS S3 | Presigned URLs, lifecycle policies |
| Auth | API key in header | Simple, industry standard |
| Errors | Fail fast, no retries | Client controls retry logic |

## Cost Per Request

| Operation | GPU | Duration | Cost |
|-----------|-----|----------|------|
| Shape generation | A10G ($1.10/hr) | ~30s | $0.009 |
| Texture generation | L40S ($1.70/hr) | ~60s | $0.028 |
| **Total per request** | | ~90s | **~$0.037** |

Plus S3: ~$0.023/GB storage, ~$0.09/GB egress.

At scale, 1000 requests costs about $37 in compute.

## Quick Start

```bash
# Install Modal CLI
pip install modal
modal setup

# Configure secrets (see docs/DEPLOYMENT.md for details)
modal secret create aws-credentials AWS_ACCESS_KEY_ID="..." AWS_SECRET_ACCESS_KEY="..." AWS_REGION="us-east-1" S3_BUCKET_NAME="your-bucket"
modal secret create api-keys VALID_API_KEYS="sk_live_yourkey"

# Deploy
modal deploy modal_app/main.py
```

Then call the API:

```bash
curl -X POST https://your-app.modal.run/generate/stream \
  -H "X-API-Key: sk_live_yourkey" \
  -F "image=@photo.png"
```

## Documentation

| Document | Purpose |
|----------|---------|
| [docs/API.md](docs/API.md) | API reference, endpoints, SSE events |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Secrets, deployment, monitoring, troubleshooting |
| [examples/README.md](examples/README.md) | Python client, Gradio app, mesh utilities |

## Module Structure

```
modal_app/
├── main.py              # Modal app entry point
├── image.py             # Container image definition
├── config.py            # Configuration
├── services/            # GPU services (shape, texture)
├── storage/             # S3 integration
├── api/                 # FastAPI routes, auth, SSE
├── examples/            # Client implementations
├── docs/                # API and deployment guides
└── tests/               # Unit and integration tests
```

## What's Not Included

The Modal API focuses on image-to-3D generation. These features from the
original Hunyuan3D-2.1 are not available:

- **Text-to-3D**: Use an external image generator, then pass the image here
- **Multi-view input**: Single image only
- **Custom model weights**: Official pretrained weights only

## Running Tests

```bash
# Unit tests (fast, no GPU)
pytest modal_app/tests/ -v

# Integration tests (requires Modal deployment)
modal run modal_app/tests/integration/test_shape_generator_modal.py
```
