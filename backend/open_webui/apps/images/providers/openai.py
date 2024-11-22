# backend/open_webui/apps/images/providers/openai.py

import logging
from typing import List, Dict, Optional

import httpx
from open_webui.config import IMAGES_OPENAI_API_BASE_URL, IMAGES_OPENAI_API_KEY
from .base import BaseImageProvider
from .registry import provider_registry

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
            base_url=IMAGES_OPENAI_API_BASE_URL.value,
            api_key=IMAGES_OPENAI_API_KEY.value,
        )
        log.debug(f"OpenAIProvider initialized with base_url: {self.base_url}")

    def populate_config(self):
        """
        Populate the shared configuration with OpenAI-specific details.
        """
        IMAGES_OPENAI_API_BASE_URL.value = self.base_url
        IMAGES_OPENAI_API_KEY.value = self.api_key

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
                    image_filename = self.save_b64_image(b64_image)
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

    async def verify_url(self):
        """
        Verify the connectivity of OpenAI's API endpoint.
        """
        try:
            async with httpx.AsyncClient() as client:
                # OpenAI doesn't have a specific status endpoint for image generation.
                # We'll perform a simple request to list available models as a connectivity check.
                response = await client.get(
                    url=f"{self.base_url}/models",
                    headers=self.headers,
                    timeout=10.0,
                )
            response.raise_for_status()
            models = response.json()
            log.info(f"OpenAI API is reachable. Retrieved models: {models}")
        except Exception as e:
            log.error(f"Failed to verify OpenAI API: {e}")
            raise Exception(f"Failed to verify OpenAI API: {e}")

    def get_config(self) -> Dict[str, Optional[str]]:
        """
        Retrieve OpenAI-specific configuration details.

        Returns:
            Dict[str, Optional[str]]: OpenAI configuration details.
        """
        return {
            "IMAGES_OPENAI_API_BASE_URL": self.base_url,
            "IMAGES_OPENAI_API_KEY": self.api_key,
        }


# Register the provider
provider_registry.register("openai", OpenAIProvider)
