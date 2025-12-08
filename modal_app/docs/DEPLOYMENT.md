# Hunyuan3D-2.1 Deployment Guide

## Prerequisites

### Accounts

1. **Modal Account**
   - Sign up at https://modal.com
   - GPU access enabled (A10G and L40S required)
   - Install CLI: `pip install modal`
   - Authenticate: `modal setup`

2. **AWS Account**
   - S3 bucket created for artifact storage
   - IAM user with S3 access

3. **HuggingFace Account** (build time only)
   - For downloading model weights

### Required IAM Policy

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject"
      ],
      "Resource": "arn:aws:s3:::YOUR-BUCKET-NAME/results/*"
    }
  ]
}
```

## Modal Secrets Configuration

### AWS Credentials

```bash
modal secret create aws-credentials \
    AWS_ACCESS_KEY_ID="AKIA..." \
    AWS_SECRET_ACCESS_KEY="..." \
    AWS_REGION="us-east-1" \
    S3_BUCKET_NAME="your-bucket-name"
```

### API Keys

Generate secure keys:

```python
import secrets
print(f"sk_live_{secrets.token_urlsafe(24)}")
print(f"sk_test_{secrets.token_urlsafe(24)}")
```

Create the secret:

```bash
modal secret create api-keys \
    VALID_API_KEYS="sk_live_xxx,sk_test_yyy"
```

## S3 Lifecycle Policy

Configure automatic cleanup of old artifacts:

```bash
aws s3api put-bucket-lifecycle-configuration \
    --bucket YOUR-BUCKET-NAME \
    --lifecycle-configuration '{
      "Rules": [{
        "ID": "DeleteOldArtifacts",
        "Filter": {"Prefix": "results/"},
        "Status": "Enabled",
        "Expiration": {"Days": 2}
      }]
    }'
```

## Deployment

### Development (hot-reload)

```bash
modal serve modal_app/main.py
```

This starts the API locally with hot-reload enabled.

### Production

```bash
modal deploy modal_app/main.py
```

This deploys to Modal's infrastructure with the URL:
```
https://your-workspace--hunyuan3d-api-api.modal.run
```

## Verification

### Health Check

```bash
curl https://your-app.modal.run/health
```

### Test Generation

```bash
curl -X POST https://your-app.modal.run/generate/stream \
    -H "X-API-Key: sk_test_xxx" \
    -F "image=@test.png" \
    -F "generate_texture=false"
```

## Monitoring

### Logs

View logs in the Modal dashboard or via CLI:

```bash
modal app logs hunyuan3d-api
```

### Metrics

Modal provides built-in metrics:
- Container cold starts
- Request latency
- GPU utilization
- Memory usage

Access via Modal dashboard: https://modal.com/apps

## Scaling

Modal automatically scales containers based on demand:
- Minimum: 0 containers (scale to zero when idle)
- Maximum: Determined by your Modal plan
- Scale-down window: 300 seconds (5 minutes)

No manual configuration required.

## Cost Projections

| Operation | GPU | Duration | Cost |
|-----------|-----|----------|------|
| Shape generation | A10G ($1.10/hr) | ~30s | $0.009 |
| Texture generation | L40S ($1.70/hr) | ~60s | $0.028 |
| **Total per request** | | ~90s | **~$0.037** |

S3 costs (approximate):
- Storage: $0.023/GB/month
- Egress: $0.09/GB

## Updating API Keys

To rotate API keys:

```bash
# Generate new keys
python -c "import secrets; print(f'sk_live_{secrets.token_urlsafe(24)}')"

# Update the secret (add new key, keep old temporarily)
modal secret create api-keys --force \
    VALID_API_KEYS="new_key,old_key"

# After clients migrate, remove old key
modal secret create api-keys --force \
    VALID_API_KEYS="new_key"
```

## Rollback

To rollback to a previous version:

```bash
# List deployments
modal app history hunyuan3d-api

# Rollback (if needed - redeploy previous code)
git checkout <previous-commit>
modal deploy modal_app/main.py
```

## Troubleshooting

### Container fails to start

1. Check logs: `modal app logs hunyuan3d-api`
2. Verify secrets exist: `modal secret list`
3. Test image build: `modal run modal_app/smoke_test.py`

### GPU out of memory

1. Check concurrent requests
2. Consider reducing batch size (not currently exposed)
3. Contact Modal support for larger GPU options

### S3 upload failures

1. Verify AWS credentials: `modal secret list`
2. Check bucket permissions
3. Verify bucket exists in correct region

### Slow cold starts

Cold starts are expected (~20s with memory snapshot). To minimize:
- Keep containers warm with periodic health checks
- Consider Modal's "keep warm" feature for production
