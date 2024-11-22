# backend/open_webui/apps/images/providers/togetherai.py

import json
import logging
from typing import List, Dict, Optional

import httpx
from open_webui.config import IMAGES_TOGETHERAI_BASE_URL, IMAGES_TOGETHERAI_API_KEY
from .base import BaseImageProvider
from .registry import provider_registry

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
            base_url=IMAGES_TOGETHERAI_BASE_URL.value,
            api_key=IMAGES_TOGETHERAI_API_KEY.value,
        )
        log.debug(f"TogetherAIProvider initialized with base_url: {self.base_url}")

    def populate_config(self):
        """
        Populate the shared configuration with TogetherAI-specific details.
        """
        IMAGES_TOGETHERAI_BASE_URL.value = self.base_url
        IMAGES_TOGETHERAI_API_KEY.value = self.api_key

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
            "steps": int(self.additional_headers.get("steps", 50)),
            "n": n,
            "negative_prompt": negative_prompt or "",
            "response_format": "b64_json",
        }

        log.debug(f"TogetherAIProvider Payload: {json.dumps(payload, indent=2)}")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url=f"{self.base_url}/generate",
                    headers=self.headers,
                    json=payload,
                    timeout=120.0,
                )
            log.debug(f"TogetherAIProvider Response Status: {response.status_code}")
            response.raise_for_status()
            res = response.json()
            log.debug(f"TogetherAIProvider Response: {json.dumps(res, indent=2)}")

            images = []
            for img in res.get("images", []):
                b64_image = img.get("b64_json")
                if b64_image:
                    image_filename = self.save_b64_image(b64_image)
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
            log.debug(f"TogetherAIProvider Models Response: {json.dumps(models, indent=2)}")
            return [{"id": model.get("id", "unknown"), "name": model.get("name", "unknown")} for model in models]
        except Exception as e:
            log.error(f"Error listing TogetherAI models: {e}")
            return []

    def get_config(self) -> Dict[str, Optional[str]]:
        """
        Retrieve TogetherAI-specific configuration details.

        Returns:
            Dict[str, Optional[str]]: TogetherAI configuration details.
        """
        return {
            "IMAGES_TOGETHERAI_BASE_URL": self.base_url,
            "IMAGES_TOGETHERAI_API_KEY": self.api_key,
        }


# Register the provider
provider_registry.register("togetherai", TogetherAIProvider)
