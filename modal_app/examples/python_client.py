#!/usr/bin/env python3
"""Example Python client for the Hunyuan3D API.

This client demonstrates how to:
1. Send a generation request
2. Parse SSE events as they arrive
3. Display progress to the user
4. Download the result when complete
5. Handle errors gracefully

Usage:
    python modal_app/examples/python_client.py --image photo.png --api-key sk_live_xxx

Requirements:
    pip install httpx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from modal_app.api.sse import parse_sse_line


def generate_3d_mesh(
    api_url: str,
    api_key: str,
    image_path: Path,
    generate_texture: bool = True,
    seed: int | None = None,
) -> str | None:
    """Generate a 3D mesh from an image.

    Args:
        api_url: Base URL of the Hunyuan3D API
        api_key: Valid API key
        image_path: Path to input image
        generate_texture: Whether to generate PBR textures
        seed: Optional random seed for reproducibility

    Returns:
        Download URL on success, None on failure
    """
    import httpx

    # Prepare request
    files = {"image": image_path.open("rb")}
    data = {"generate_texture": str(generate_texture).lower()}
    if seed is not None:
        data["seed"] = str(seed)

    headers = {"X-API-Key": api_key}

    print(f"Sending request to {api_url}/generate/stream...")
    print(f"  Image: {image_path}")
    print(f"  Texture: {generate_texture}")
    if seed:
        print(f"  Seed: {seed}")
    print()

    # Make streaming request
    with httpx.Client(timeout=300) as client:
        with client.stream(
            "POST",
            f"{api_url}/generate/stream",
            headers=headers,
            files=files,
            data=data,
        ) as response:
            if response.status_code != 200:
                print(f"Error: HTTP {response.status_code}")
                print(response.text)
                return None

            current_event = None
            download_url = None

            # Process SSE events line by line
            for line in response.iter_lines():
                line_type, value = parse_sse_line(line)

                if line_type == "event":
                    current_event = value
                elif line_type == "data" and current_event:
                    # Handle different event types
                    if current_event == "started":
                        print(f"Started: job_id={value.get('job_id')}")

                    elif current_event == "progress":
                        stage = value.get("stage", "unknown")
                        percent = value.get("percent", 0)
                        bar = "=" * (percent // 5) + ">" + " " * (20 - percent // 5)
                        print(
                            f"\r  [{bar}] {percent:3d}% - {stage}", end="", flush=True
                        )

                    elif current_event == "completed":
                        print()  # Newline after progress
                        download_url = value.get("download_url")
                        print(f"Completed! Download URL: {download_url}")

                    elif current_event == "error":
                        print()  # Newline after progress
                        print(f"Error: {value.get('message')}")
                        print(f"  Stage: {value.get('stage')}")
                        print(f"  Retriable: {value.get('retriable')}")
                        return None

                    current_event = None

    return download_url


def download_result(url: str, output_path: Path) -> bool:
    """Download the generated GLB file.

    Args:
        url: Presigned S3 URL
        output_path: Where to save the file

    Returns:
        True on success, False on failure
    """
    import httpx

    print(f"Downloading to {output_path}...")

    try:
        with httpx.Client(timeout=60) as client:
            response = client.get(url)
            response.raise_for_status()

            output_path.write_bytes(response.content)
            print(f"Saved {len(response.content):,} bytes to {output_path}")
            return True

    except httpx.HTTPError as e:
        print(f"Download failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate 3D mesh from image")
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument("--api-key", required=True, help="API key")
    parser.add_argument(
        "--api-url",
        default="https://your-app.modal.run",
        help="API base URL",
    )
    parser.add_argument(
        "--no-texture",
        action="store_true",
        help="Skip texture generation (shape only)",
    )
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output.glb"),
        help="Output file path",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Don't download the result",
    )

    args = parser.parse_args()

    if not args.image.exists():
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    # Generate
    download_url = generate_3d_mesh(
        api_url=args.api_url,
        api_key=args.api_key,
        image_path=args.image,
        generate_texture=not args.no_texture,
        seed=args.seed,
    )

    if not download_url:
        sys.exit(1)

    # Download (optional)
    if not args.no_download:
        if not download_result(download_url, args.output):
            sys.exit(1)

    print("\nDone!")


if __name__ == "__main__":
    main()
