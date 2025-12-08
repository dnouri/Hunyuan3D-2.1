"""Generate test fixture files.

This script creates minimal but valid fixture files for testing:
- PNG images (1x1, 256x256)
- GLB mesh (minimal valid structure)

Run once to generate fixtures, then commit the results.
"""

import json
import struct
from pathlib import Path


def generate_minimal_png(width: int = 1, height: int = 1) -> bytes:
    """Generate a minimal valid PNG image.

    Creates a solid color PNG of specified dimensions.
    """
    import zlib

    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk_len = struct.pack(">I", len(data))
        chunk_crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        return chunk_len + chunk_type + data + chunk_crc

    signature = b"\x89PNG\r\n\x1a\n"

    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = png_chunk(b"IHDR", ihdr_data)

    raw_data = b""
    for _ in range(height):
        raw_data += b"\x00"
        raw_data += b"\x80\x80\x80" * width

    compressed = zlib.compress(raw_data)
    idat = png_chunk(b"IDAT", compressed)

    iend = png_chunk(b"IEND", b"")

    return signature + ihdr + idat + iend


def generate_minimal_glb() -> bytes:
    """Generate a minimal valid GLB (glTF binary) file.

    Creates a GLB with a single triangle mesh.
    """
    gltf_json = {
        "asset": {"version": "2.0", "generator": "test-fixture"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0},
                        "indices": 1,
                    }
                ]
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "max": [1.0, 1.0, 0.0],
                "min": [0.0, 0.0, 0.0],
            },
            {
                "bufferView": 1,
                "componentType": 5123,
                "count": 3,
                "type": "SCALAR",
            },
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36},
            {"buffer": 0, "byteOffset": 36, "byteLength": 6},
        ],
        "buffers": [{"byteLength": 44}],
    }

    positions = struct.pack("<9f", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0)
    indices = struct.pack("<3H", 0, 1, 2)
    padding = b"\x00\x00"

    binary_data = positions + indices + padding

    json_str = json.dumps(gltf_json, separators=(",", ":"))
    json_bytes = json_str.encode("utf-8")

    json_padding = (4 - len(json_bytes) % 4) % 4
    json_bytes += b" " * json_padding

    bin_padding = (4 - len(binary_data) % 4) % 4
    binary_data += b"\x00" * bin_padding

    total_length = 12 + 8 + len(json_bytes) + 8 + len(binary_data)

    header = struct.pack("<4sII", b"glTF", 2, total_length)
    json_chunk_header = struct.pack("<II", len(json_bytes), 0x4E4F534A)
    bin_chunk_header = struct.pack("<II", len(binary_data), 0x004E4942)

    return header + json_chunk_header + json_bytes + bin_chunk_header + binary_data


def main():
    fixtures_dir = Path(__file__).parent

    png_1x1 = generate_minimal_png(1, 1)
    (fixtures_dir / "test_image_1x1.png").write_bytes(png_1x1)
    print(f"Generated test_image_1x1.png ({len(png_1x1)} bytes)")

    png_256 = generate_minimal_png(256, 256)
    (fixtures_dir / "test_image_256x256.png").write_bytes(png_256)
    print(f"Generated test_image_256x256.png ({len(png_256)} bytes)")

    glb = generate_minimal_glb()
    (fixtures_dir / "test_mesh.glb").write_bytes(glb)
    print(f"Generated test_mesh.glb ({len(glb)} bytes)")


if __name__ == "__main__":
    main()
