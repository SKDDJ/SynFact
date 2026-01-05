"""Abstract image generation client for SynFact-VL."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ImageConfig(BaseModel):
    """Configuration for image generation API."""

    provider: str = Field(default="mock", description="Image provider: mock, nano_banana, flux, qwen")
    base_url: str = Field(default="", description="API base URL")
    api_key: str = Field(default="", description="API key")
    model_name: str = Field(default="", description="Model name for the provider")
    output_format: str = Field(default="png", description="Output image format")
    default_width: int = Field(default=1024, description="Default image width")
    default_height: int = Field(default=1024, description="Default image height")


class ImageGenerationResult(BaseModel):
    """Result of image generation."""

    success: bool
    image_bytes: bytes | None = None
    image_path: str | None = None
    error_message: str | None = None
    prompt_used: str = ""


class ImageClient(Protocol):
    """Protocol for image generation clients."""

    def generate(
        self,
        prompt: str,
        width: int | None = None,
        height: int | None = None,
        **kwargs,
    ) -> ImageGenerationResult:
        """Generate an image from a text prompt.

        Args:
            prompt: Text description of the image to generate.
            width: Optional width override.
            height: Optional height override.
            **kwargs: Additional provider-specific parameters.

        Returns:
            ImageGenerationResult with image bytes or error.
        """
        ...


class MockImageClient:
    """Mock image client for testing without actual image generation."""

    def __init__(self, config: ImageConfig):
        """Initialize mock client.

        Args:
            config: Image configuration.
        """
        self.config = config
        logger.info("Initialized MockImageClient (no actual images will be generated)")

    def generate(
        self,
        prompt: str,
        width: int | None = None,
        height: int | None = None,
        **kwargs,
    ) -> ImageGenerationResult:
        """Return a placeholder result without generating an image.

        Args:
            prompt: Text description (logged but not used).
            width: Ignored.
            height: Ignored.

        Returns:
            ImageGenerationResult with placeholder path.
        """
        logger.info(f"[MOCK] Would generate image for prompt: {prompt[:100]}...")

        # Create a simple 1x1 pixel PNG as placeholder
        # PNG header + minimal IHDR + IDAT + IEND
        placeholder_png = (
            b'\x89PNG\r\n\x1a\n'  # PNG signature
            b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde'
            b'\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N'
            b'\x00\x00\x00\x00IEND\xaeB`\x82'
        )

        return ImageGenerationResult(
            success=True,
            image_bytes=placeholder_png,
            prompt_used=prompt,
        )


class NanoBananaClient:
    """Nano-banana API client for image generation."""

    def __init__(self, config: ImageConfig):
        """Initialize Nano-banana client.

        Args:
            config: Image configuration with API credentials.
        """
        self.config = config
        self.base_url = config.base_url
        self.api_key = config.api_key
        logger.info(f"Initialized NanoBananaClient with base_url: {self.base_url}")

    def generate(
        self,
        prompt: str,
        width: int | None = None,
        height: int | None = None,
        **kwargs,
    ) -> ImageGenerationResult:
        """Generate image using Nano-banana API.

        Args:
            prompt: Text description of the image.
            width: Image width.
            height: Image height.

        Returns:
            ImageGenerationResult with image bytes.
        """
        import httpx

        width = width or self.config.default_width
        height = height or self.config.default_height

        try:
            # This is a placeholder implementation - adjust based on actual API
            response = httpx.post(
                f"{self.base_url}/generate",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "model": self.config.model_name,
                    **kwargs,
                },
                timeout=120.0,
            )
            response.raise_for_status()

            return ImageGenerationResult(
                success=True,
                image_bytes=response.content,
                prompt_used=prompt,
            )

        except Exception as e:
            logger.error(f"Nano-banana generation failed: {e}")
            return ImageGenerationResult(
                success=False,
                error_message=str(e),
                prompt_used=prompt,
            )


class LocalFluxClient:
    """Local Flux model client for image generation."""

    def __init__(self, config: ImageConfig):
        """Initialize local Flux client.

        Args:
            config: Image configuration with model path.
        """
        self.config = config
        self.model_path = config.base_url  # Reuse base_url for model path
        self._model = None
        logger.info(f"Initialized LocalFluxClient with model_path: {self.model_path}")

    def _load_model(self):
        """Lazy load the Flux model."""
        if self._model is None:
            # Placeholder - implement actual model loading
            logger.info("Loading Flux model...")
            # from diffusers import FluxPipeline
            # self._model = FluxPipeline.from_pretrained(self.model_path)
            raise NotImplementedError("LocalFluxClient requires diffusers setup")

    def generate(
        self,
        prompt: str,
        width: int | None = None,
        height: int | None = None,
        **kwargs,
    ) -> ImageGenerationResult:
        """Generate image using local Flux model.

        Args:
            prompt: Text description.
            width: Image width.
            height: Image height.

        Returns:
            ImageGenerationResult with image bytes.
        """
        try:
            self._load_model()
            # Placeholder implementation
            # image = self._model(prompt, width=width, height=height).images[0]
            # buffer = io.BytesIO()
            # image.save(buffer, format="PNG")
            # return ImageGenerationResult(success=True, image_bytes=buffer.getvalue())
            raise NotImplementedError("LocalFluxClient not fully implemented")

        except Exception as e:
            logger.error(f"Local Flux generation failed: {e}")
            return ImageGenerationResult(
                success=False,
                error_message=str(e),
                prompt_used=prompt,
            )


def create_image_client(config: ImageConfig) -> ImageClient:
    """Factory function to create the appropriate image client.

    Args:
        config: Image configuration.

    Returns:
        An ImageClient implementation.
    """
    provider = config.provider.lower()

    if provider == "mock":
        return MockImageClient(config)
    elif provider == "nano_banana":
        return NanoBananaClient(config)
    elif provider == "flux":
        return LocalFluxClient(config)
    else:
        logger.warning(f"Unknown provider '{provider}', falling back to mock")
        return MockImageClient(config)


def save_image(
    image_bytes: bytes,
    output_dir: Path,
    filename: str,
) -> Path:
    """Save image bytes to file.

    Args:
        image_bytes: Raw image data.
        output_dir: Directory to save image.
        filename: Filename without extension.

    Returns:
        Path to saved image.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = output_dir / f"{filename}.png"
    with open(image_path, "wb") as f:
        f.write(image_bytes)

    logger.debug(f"Saved image to {image_path}")
    return image_path
