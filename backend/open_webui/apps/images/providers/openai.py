import logging
from typing import List, Dict, Optional

import httpx

from .base import BaseImageProvider
from .registry import provider_registry
from open_webui.config import IMAGES_OPENAI_API_BASE_URL, IMAGES_OPENAI_API_KEY

log = logging.getLogger(__name__)


class OpenAIProvider(BaseImageProvider):
    """
    Provider for OpenAI's DALL·E image generation API.
    """

    def __init__(self):
        """
        Initialize the OpenAI provider with its specific configuration.
        """
        super().__init__(
            base_url=str(IMAGES_OPENAI_API_BASE_URL.value),
            api_key=str(IMAGES_OPENAI_API_KEY.value),
            additional_headers={},
        )

    async def generate_image(
        self, prompt: str, n: int, size: str, negative_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Generate images using OpenAI's DALL·E API.

        Args:
            prompt (str): The text prompt for image generation.
            n (int): Number of images to generate.
            size (str): Dimensions of the image (e.g., "512x512").
            negative_prompt (Optional[str]): Ignored for OpenAI, kept for compatibility.

        Returns:
            List[Dict[str, str]]: List of URLs pointing to generated images.
        """
        payload = {
            "model": "dall-e-2",  # Default model
            "prompt": prompt,
            "n": n,
            "size": size,
            "response_format": "b64_json",
        }

        log.debug(f"OpenAIProvider Payload: {payload}")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url=f"{self.base_url}/images/generations",
                    headers=self.headers,
                    json=payload,
                    timeout=120.0,
                )
            log.debug(f"OpenAIProvider Response Status: {response.status_code}")
            response.raise_for_status()
            res = response.json()
            log.debug(f"OpenAIProvider Response: {res}")

            images = []
            for image in res.get("data", []):
                b64_image = image.get("b64_json")
                if b64_image:
                    image_filename = await self.save_b64_image(b64_image)
                    if image_filename:
                        images.append({"url": f"/cache/image/generations/{image_filename}"})
            return images

        except httpx.RequestError as e:
            log.error(f"OpenAIProvider Request failed: {e}")
            raise Exception(f"OpenAIProvider Request failed: {e}")
        except Exception as e:
            log.error(f"OpenAIProvider Error: {e}")
            raise Exception(f"OpenAIProvider Error: {e}")

    async def list_models(self) -> List[Dict[str, str]]:
        """
        List available models for OpenAI's DALL·E API.

        Returns:
            List[Dict[str, str]]: List of available models.
        """
        # OpenAI's image generation currently only supports DALL·E models
        return [
            {"id": "dall-e-2", "name": "DALL·E 2"},
            {"id": "dall-e-3", "name": "DALL·E 3"},
        ]

    def get_config(self) -> Dict[str, str]:
        """
        Retrieve OpenAI-specific configuration details.

        Returns:
            Dict[str, str]: OpenAI configuration details.
        """
        return {
            "OPENAI_API_BASE_URL": str(IMAGES_OPENAI_API_BASE_URL.value),
            "OPENAI_API_KEY": str(IMAGES_OPENAI_API_KEY.value),
        }


# Register the provider
provider_registry.register("openai", OpenAIProvider)
