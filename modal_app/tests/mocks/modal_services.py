"""Mock implementations for Modal GPU services.

These mocks replicate the interface and behavior of real Modal services.
They support:
- Configurable delays to simulate GPU processing time
- Realistic return types (bytes for meshes, dicts for metadata)
- Error simulation for testing error handling
- Modal's `.remote()` call pattern

Usage in tests:
    from modal_app.tests.mocks.modal_services import MockShapeGenerator, MockTextureGenerator

    @pytest.fixture
    def shape_generator():
        return MockShapeGenerator(delay=0.1)  # Fast for unit tests

    # Use just like real Modal services:
    result = shape_generator.generate.remote(image_bytes)
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


def _load_fixture(name: str) -> bytes:
    """Load a fixture file from the fixtures directory."""
    fixtures_dir = Path(__file__).parent.parent / "fixtures"
    return (fixtures_dir / name).read_bytes()


class ModalMethod(Generic[T]):
    """Simulates Modal's method wrapper with .remote() pattern.

    Real Modal usage:
        result = service.method.remote(args)

    This wrapper provides the same interface for mocks.
    """

    def __init__(self, func: Callable[..., T]) -> None:
        self._func = func

    def __call__(self, *args, **kwargs) -> T:
        """Direct call (local execution in Modal terms)."""
        return self._func(*args, **kwargs)

    def remote(self, *args, **kwargs) -> T:
        """Remote call (simulates Modal's .remote() pattern)."""
        return self._func(*args, **kwargs)

    def spawn(self, *args, **kwargs) -> "MockFunctionCall[T]":
        """Async spawn (simulates Modal's .spawn() pattern)."""
        return MockFunctionCall(self._func, args, kwargs)


@dataclass
class MockFunctionCall(Generic[T]):
    """Simulates Modal's FunctionCall handle from .spawn()."""

    _func: Callable[..., T]
    _args: tuple
    _kwargs: dict
    _result: T | None = field(default=None, init=False)
    _executed: bool = field(default=False, init=False)

    def get(self) -> T:
        """Block and get the result."""
        if not self._executed:
            self._result = self._func(*self._args, **self._kwargs)
            self._executed = True
        return self._result


@dataclass
class MockShapeGenerator:
    """Mock implementation of ShapeGenerator service.

    Simulates the behavior of the real ShapeGenerator running on A10G GPU:
    - Accepts image bytes as input
    - Returns GLB mesh bytes as output
    - Supports configurable delay to simulate processing time
    - Tracks call count and last input for verification

    Usage:
        generator = MockShapeGenerator(delay=0.1)
        result = generator.generate.remote(image_bytes)  # Modal pattern
    """

    delay: float = 0.0
    fail_on_call: int | None = None
    error_message: str = "Simulated GPU error"

    _call_count: int = field(default=0, init=False)
    _last_input: bytes | None = field(default=None, init=False)
    _last_seed: int | None = field(default=None, init=False)
    _last_steps: int | None = field(default=None, init=False)
    _last_guidance_scale: float | None = field(default=None, init=False)
    _last_octree_resolution: int | None = field(default=None, init=False)
    _mesh_bytes: bytes | None = field(default=None, init=False)
    _generate: ModalMethod | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            self._mesh_bytes = _load_fixture("test_mesh.glb")
        except FileNotFoundError:
            self._mesh_bytes = b"mock-glb-data"

        self._generate = ModalMethod(self._do_generate)

    @property
    def generate(self) -> ModalMethod:
        """Method that generates mesh from image.

        Supports Modal's call patterns:
            generator.generate.remote(image_bytes)  # Remote call
            generator.generate(image_bytes)         # Local call
            handle = generator.generate.spawn(image_bytes)  # Async
        """
        return self._generate

    def _do_generate(
        self,
        image_bytes: bytes,
        seed: int | None = None,
        steps: int = 30,
        guidance_scale: float = 5.0,
        octree_resolution: int = 256,
    ) -> bytes:
        """Internal implementation of generate()."""
        self._call_count += 1
        self._last_input = image_bytes
        self._last_seed = seed
        self._last_steps = steps
        self._last_guidance_scale = guidance_scale
        self._last_octree_resolution = octree_resolution

        if not image_bytes:
            raise ValueError("image_bytes cannot be empty")

        if len(image_bytes) < 8:
            raise ValueError("image_bytes too small to be valid image")

        if self.fail_on_call == self._call_count:
            raise RuntimeError(self.error_message)

        if self.delay > 0:
            time.sleep(self.delay)

        return self._mesh_bytes

    @property
    def call_count(self) -> int:
        """Number of times generate() was called."""
        return self._call_count

    @property
    def last_input(self) -> bytes | None:
        """The last image_bytes passed to generate()."""
        return self._last_input

    @property
    def last_seed(self) -> int | None:
        """The last seed passed to generate()."""
        return self._last_seed

    @property
    def last_steps(self) -> int | None:
        """The last steps passed to generate()."""
        return self._last_steps

    @property
    def last_guidance_scale(self) -> float | None:
        """The last guidance_scale passed to generate()."""
        return self._last_guidance_scale

    @property
    def last_octree_resolution(self) -> int | None:
        """The last octree_resolution passed to generate()."""
        return self._last_octree_resolution


@dataclass
class MockTextureGenerator:
    """Mock implementation of TextureGenerator service.

    Simulates the behavior of the real TextureGenerator running on L40S GPU:
    - Accepts mesh bytes and image bytes as input
    - Returns textured GLB mesh bytes as output
    - Supports configurable delay to simulate processing time

    Usage:
        generator = MockTextureGenerator(delay=0.1)
        result = generator.generate.remote(mesh_bytes, image_bytes)  # Modal pattern
    """

    delay: float = 0.0
    fail_on_call: int | None = None
    error_message: str = "Simulated GPU error"

    _call_count: int = field(default=0, init=False)
    _last_mesh: bytes | None = field(default=None, init=False)
    _last_image: bytes | None = field(default=None, init=False)
    _textured_mesh_bytes: bytes | None = field(default=None, init=False)
    _generate: ModalMethod | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            self._textured_mesh_bytes = _load_fixture("test_mesh.glb")
        except FileNotFoundError:
            self._textured_mesh_bytes = b"mock-textured-glb-data"

        self._generate = ModalMethod(self._do_generate)

    @property
    def generate(self) -> ModalMethod:
        """Method that generates textured mesh.

        Supports Modal's call patterns:
            generator.generate.remote(mesh_bytes, image_bytes)
            generator.generate(mesh_bytes, image_bytes)
            handle = generator.generate.spawn(mesh_bytes, image_bytes)
        """
        return self._generate

    def _do_generate(
        self,
        mesh_bytes: bytes,
        image_bytes: bytes,
        seed: int | None = None,
    ) -> bytes:
        """Internal implementation of generate()."""
        self._call_count += 1
        self._last_mesh = mesh_bytes
        self._last_image = image_bytes

        if not mesh_bytes:
            raise ValueError("mesh_bytes cannot be empty")

        if not image_bytes:
            raise ValueError("image_bytes cannot be empty")

        if self.fail_on_call == self._call_count:
            raise RuntimeError(self.error_message)

        if self.delay > 0:
            time.sleep(self.delay)

        return self._textured_mesh_bytes

    @property
    def call_count(self) -> int:
        """Number of times generate() was called."""
        return self._call_count


@dataclass
class MockS3Storage:
    """Mock implementation of S3 storage.

    Simulates S3 upload and presigned URL generation without AWS.
    Uses moto for more realistic testing in integration tests.
    """

    bucket_name: str = "mock-bucket"
    region: str = "us-east-1"
    presigned_url_base: str = "https://mock-bucket.s3.amazonaws.com"

    _uploads: dict = field(default_factory=dict, init=False)

    def upload_artifact(
        self,
        job_id: str,
        data: bytes,
        content_type: str = "model/gltf-binary",
    ) -> str:
        """Upload artifact and return presigned URL.

        Args:
            job_id: Unique identifier for the job
            data: Binary data to upload
            content_type: MIME type of the data

        Returns:
            Presigned URL for downloading the artifact
        """
        key = f"results/{job_id}.glb"
        self._uploads[key] = {
            "data": data,
            "content_type": content_type,
        }
        return self.generate_presigned_url(job_id)

    def generate_presigned_url(self, job_id: str, expires_in: int = 3600) -> str:
        """Generate a presigned URL for an artifact.

        Args:
            job_id: Unique identifier for the job
            expires_in: URL expiration time in seconds

        Returns:
            Presigned URL string
        """
        return f"{self.presigned_url_base}/results/{job_id}.glb?mock-signature=xxx"

    def get_upload(self, job_id: str) -> bytes | None:
        """Retrieve uploaded data for verification in tests."""
        key = f"results/{job_id}.glb"
        upload = self._uploads.get(key)
        return upload["data"] if upload else None
