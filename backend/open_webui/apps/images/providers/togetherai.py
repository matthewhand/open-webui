import logging
from typing import List, Dict, Optional

import httpx

from .base import BaseImageProvider
from .registry import provider_registry
from open_webui.config import (
    IMAGES_TOGETHERAI_BASE_URL,
    IMAGES_TOGETHERAI_API_KEY,
)

log = logging.getLogger(__name__)


class TogetherAIProvider(BaseImageProvider):
    """
    Provider for TogetherAI-based image generation.
    """

    def __init__(self):
        """
        Initialize TogetherAI provider with its specific configuration.
        """
        super().__init__(
            base_url=str(IMAGES_TOGETHERAI_BASE_URL.value),
            api_key=str(IMAGES_TOGETHERAI_API_KEY.value),
            additional_headers={},  # Add any additional headers if required
        )

    async def generate_image(
        self, prompt: str, n: int, size: str, negative_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Generate images using TogetherAI's API.

        Args:
            prompt (str): The text prompt for image generation.
            n (int): Number of images to generate.
            size (str): Dimensions of the image (e.g., "512x512").
            negative_prompt (Optional[str]): Text to exclude from the generated images.

        Returns:
            List[Dict[str, str]]: List of URLs pointing to generated images.
        """
        width, height = map(int, size.lower().split("x"))
        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": self.additional_headers.get("steps", 50),
            "n": n,
            "response_format": "b64_json",
        }

        log.debug(f"TogetherAIProvider Payload: {payload}")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url=self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=120.0,
                )
            log.debug(f"TogetherAIProvider Response Status: {response.status_code}")
            response.raise_for_status()
            res = response.json()
            log.debug(f"TogetherAIProvider Response: {res}")

            images = []
            for img in res.get("images", []):
                b64_image = img.get("b64_json")
                if b64_image:
                    image_filename = await self.save_b64_image(b64_image)
                    if image_filename:
                        images.append({"url": f"/cache/image/generations/{image_filename}"})
            return images

        except httpx.RequestError as e:
            log.error(f"TogetherAIProvider Request failed: {e}")
            raise Exception(f"TogetherAIProvider Request failed: {e}")
        except Exception as e:
            log.error(f"TogetherAIProvider Error: {e}")
            raise Exception(f"TogetherAIProvider Error: {e}")

    async def list_models(self) -> List[Dict[str, str]]:
        """
        List available models for image generation from TogetherAI's API.

        Returns:
            List[Dict[str, str]]: List of available models.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url=f"{self.base_url}/models",
                    headers=self.headers,
                    timeout=30.0,
                )
            response.raise_for_status()
            models = response.json()
            return [{"id": model["id"], "name": model["name"]} for model in models]
        except Exception as e:
            log.error(f"Error listing TogetherAI models: {e}")
            return []

    def get_config(self) -> Dict[str, str]:
        """
        Retrieve TogetherAI-specific configuration details.

        Returns:
            Dict[str, str]: Configuration details specific to TogetherAI.
        """
        return {
            "base_url": str(IMAGES_TOGETHERAI_BASE_URL.value),
            "api_key": str(IMAGES_TOGETHERAI_API_KEY.value),
        }


# Register the provider
provider_registry.register("togetherai", TogetherAIProvider)
