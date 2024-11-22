import json
import logging
from typing import List, Dict, Optional

import httpx

from .base import BaseImageProvider
from .registry import provider_registry
from open_webui.config import (
    IMAGES_HUGGINGFACE_BASE_URL,
    IMAGES_HUGGINGFACE_API_KEY,
    IMAGES_HUGGINGFACE_ADDITIONAL_HEADERS,
)

log = logging.getLogger(__name__)


class HuggingfaceProvider(BaseImageProvider):
    """
    Provider for Huggingface-based image generation.
    """

    def __init__(self):
        """
        Initialize the Huggingface provider with its specific configuration.
        """
        try:
            additional_headers = json.loads(IMAGES_HUGGINGFACE_ADDITIONAL_HEADERS.value)
        except json.JSONDecodeError:
            log.error("Failed to parse Huggingface additional headers. Defaulting to empty headers.")
            additional_headers = {}

        super().__init__(
            base_url=str(IMAGES_HUGGINGFACE_BASE_URL.value),
            api_key=str(IMAGES_HUGGINGFACE_API_KEY.value),
            additional_headers=additional_headers,
        )

    async def generate_image(
        self, prompt: str, n: int, size: str, negative_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Generate images using Huggingface's API.

        Args:
            prompt (str): The text prompt for image generation.
            n (int): Number of images to generate.
            size (str): Size of the image (e.g., "512x512").
            negative_prompt (Optional[str]): Negative prompt to exclude certain elements.

        Returns:
            List[Dict[str, str]]: List of generated image URLs.
        """
        width, height = map(int, size.lower().split("x"))
        payload = {
            "inputs": prompt,
            "parameters": {
                "num_inference_steps": int(self.additional_headers.get("num_inference_steps", 50)),
                "negative_prompt": negative_prompt or "",
                "width": width,
                "height": height,
                "num_images": n,
            },
        }

        log.debug(f"HuggingfaceProvider Payload: {payload}")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url=self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=120.0,
                )
            log.debug(f"HuggingfaceProvider Response Status: {response.status_code}")
            response.raise_for_status()
            res = response.json()
            log.debug(f"HuggingfaceProvider Response: {res}")

            images = []
            for img in res.get("data", []):
                if "b64_json" in img:
                    b64_image = img["b64_json"]
                    image_filename = await self.save_b64_image(b64_image)
                    if image_filename:
                        images.append({"url": f"/cache/image/generations/{image_filename}"})
            return images

        except httpx.RequestError as e:
            log.error(f"HuggingfaceProvider Request failed: {e}")
            raise Exception(f"HuggingfaceProvider Request failed: {e}")
        except Exception as e:
            log.error(f"HuggingfaceProvider Error: {e}")
            raise Exception(f"HuggingfaceProvider Error: {e}")

    async def list_models(self) -> List[Dict[str, str]]:
        """
        List available models from Huggingface.

        Returns:
            List[Dict[str, str]]: List of available models with 'id' and 'name'.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url=self.base_url,
                    headers=self.headers,
                    timeout=30.0,
                )
            response.raise_for_status()
            models = response.json()
            # Adjust based on actual response structure
            return [{"id": model["id"], "name": model["name"]} for model in models]
        except Exception as e:
            log.error(f"Error listing Huggingface models: {e}")
            return []

    def get_config(self) -> Dict[str, str]:
        """
        Retrieve Huggingface-specific configuration details.

        Returns:
            Dict[str, str]: Huggingface configuration details.
        """
        return {
            "base_url": str(IMAGES_HUGGINGFACE_BASE_URL.value),
            "api_key": str(IMAGES_HUGGINGFACE_API_KEY.value),
            "additional_headers": IMAGES_HUGGINGFACE_ADDITIONAL_HEADERS.value,
        }


# Register the provider
provider_registry.register("huggingface", HuggingfaceProvider)
