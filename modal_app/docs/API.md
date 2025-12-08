# Hunyuan3D-2.1 API Reference

## Overview

The Hunyuan3D API generates 3D meshes from images. It uses Server-Sent Events (SSE) to stream progress updates in real-time.

## Base URL

```
https://your-app.modal.run
```

## Authentication

All requests to `/generate/stream` require an API key in the `X-API-Key` header.

```bash
curl -H "X-API-Key: sk_live_xxx" ...
```

### API Key Format

- Production keys: `sk_live_` prefix
- Test keys: `sk_test_` prefix
- Minimum 32 characters recommended

## Endpoints

### Health Check

```
GET /health
```

Returns API health status. No authentication required.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-15T10:30:00.000Z"
}
```

### Generate 3D Mesh

```
POST /generate/stream
```

Generates a 3D mesh from an image with real-time progress streaming.

**Headers:**
| Header | Required | Description |
|--------|----------|-------------|
| X-API-Key | Yes | Valid API key |
| Content-Type | Yes | multipart/form-data |

**Form Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| image | file | Yes | - | Input image (PNG or JPEG) |
| generate_texture | string | No | "true" | Generate PBR textures |
| seed | string | No | - | Random seed for reproducibility |

**Response:**

Content-Type: `text/event-stream`

The response is a stream of SSE events.

## SSE Event Types

### started

Sent immediately when generation begins.

```
event: started
data: {"job_id": "550e8400-e29b-41d4-a716-446655440000"}
```

### progress

Sent during each stage of generation.

```
event: progress
data: {"stage": "shape", "percent": 10}
```

**Stages:**
| Stage | Percent | Description |
|-------|---------|-------------|
| shape | 10 | Shape generation started |
| shape_complete | 50 | Shape generation finished |
| texture | 55 | Texture generation started |
| texture_complete | 95 | Texture generation finished |
| uploading | 98 | Uploading to S3 |

### completed

Sent when generation succeeds.

```
event: completed
data: {"job_id": "...", "download_url": "https://..."}
```

The `download_url` is a presigned S3 URL valid for 1 hour.

### error

Sent when generation fails.

```
event: error
data: {"stage": "shape", "message": "GPU out of memory", "retriable": false}
```

| Field | Type | Description |
|-------|------|-------------|
| stage | string | Where the error occurred |
| message | string | Human-readable error message |
| retriable | boolean | Whether client should retry |

## Error Responses

Non-streaming errors return JSON:

### 401 Unauthorized

```json
{"detail": "Missing API key. Provide X-API-Key header."}
```

```json
{"detail": "Invalid API key."}
```

### 422 Validation Error

```json
{"detail": [{"loc": ["body", "image"], "msg": "field required"}]}
```

### 500 Internal Server Error

```json
{"detail": "API keys not configured. Set VALID_API_KEYS environment variable."}
```

## Example: curl

### Shape only

```bash
curl -X POST https://your-app.modal.run/generate/stream \
  -H "X-API-Key: sk_live_xxx" \
  -F "image=@photo.png" \
  -F "generate_texture=false"
```

### Full generation (shape + texture)

```bash
curl -X POST https://your-app.modal.run/generate/stream \
  -H "X-API-Key: sk_live_xxx" \
  -F "image=@photo.png" \
  -F "generate_texture=true" \
  -F "seed=42"
```

## Example: Python

```python
import httpx

api_key = "sk_live_xxx"
api_url = "https://your-app.modal.run"

with open("photo.png", "rb") as f:
    response = httpx.post(
        f"{api_url}/generate/stream",
        headers={"X-API-Key": api_key},
        files={"image": f},
        data={"generate_texture": "true"},
        timeout=300,
    )

# Parse SSE events
for line in response.text.split("\n"):
    if line.startswith("data: "):
        import json
        event_data = json.loads(line[6:])
        print(event_data)
```

## Rate Limits

Currently no rate limits are enforced. Future versions may add per-key rate limiting.

## Security

- API keys are validated using constant-time comparison
- AWS credentials are never exposed to clients
- Presigned URLs expire after 1 hour
- Error messages do not leak sensitive information
- All traffic is encrypted via HTTPS

## Performance

| Operation | Typical Duration |
|-----------|------------------|
| Cold start | ~20s (with memory snapshot) |
| Shape generation | ~30s |
| Texture generation | ~60s |
| S3 upload | ~2s |
| **Total (warm)** | **~95s** |
