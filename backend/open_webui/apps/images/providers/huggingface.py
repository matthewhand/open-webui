# backend/open_webui/apps/images/providers/huggingface.py

import json
import logging
from typing import List, Dict, Optional

import httpx
from open_webui.config import (
    IMAGES_HUGGINGFACE_BASE_URL,
    IMAGES_HUGGINGFACE_API_KEY,
    IMAGES_HUGGINGFACE_ADDITIONAL_HEADERS,
)
from .base import BaseImageProvider
from .registry import provider_registry

log = logging.getLogger(__name__)


class HuggingfaceProvider(BaseImageProvider):
    """
    Provider for Huggingface-based image generation.
    """

    def __init__(self):
        """
        Initialize the Huggingface provider with its specific configuration.
        """
        super().__init__(
            base_url=IMAGES_HUGGINGFACE_BASE_URL.value,
            api_key=IMAGES_HUGGINGFACE_API_KEY.value,
        )
        try:
            self.additional_headers = json.loads(IMAGES_HUGGINGFACE_ADDITIONAL_HEADERS)
        except json.JSONDecodeError:
            log.error("Failed to parse Huggingface additional headers. Defaulting to empty headers.")
            self.additional_headers = {}

        log.debug(f"HuggingfaceProvider initialized with base_url: {self.base_url}")

    def populate_config(self):
        """
        Populate the shared configuration with Huggingface-specific details.
        """
        IMAGES_HUGGINGFACE_BASE_URL.value = self.base_url
        IMAGES_HUGGINGFACE_API_KEY.value = self.api_key
        IMAGES_HUGGINGFACE_ADDITIONAL_HEADERS.value = json.dumps(self.additional_headers)

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
                "num_inference_steps": self.additional_headers.get("num_inference_steps", 50),
                "negative_prompt": negative_prompt or "",
                "width": width,
                "height": height,
                "num_images": n,
            },
        }

        log.debug(f"HuggingfaceProvider Payload: {json.dumps(payload, indent=2)}")

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
            log.debug(f"HuggingfaceProvider Response: {json.dumps(res, indent=2)}")

            images = []
            for img in res.get("data", []):
                b64_image = img.get("b64_json")
                if b64_image:
                    image_filename = self.save_b64_image(b64_image)
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
                    url=f"{self.base_url}/models",
                    headers=self.headers,
                    timeout=30.0,
                )
            response.raise_for_status()
            models = response.json()

            log.debug(f"HuggingfaceProvider Models Response: {json.dumps(models, indent=2)}")
            return [{"id": model.get("id", "unknown"), "name": model.get("name", "unknown")} for model in models]
        except Exception as e:
            log.error(f"Error listing Huggingface models: {e}")
            return []

    def get_config(self) -> Dict[str, Optional[str]]:
        """
        Retrieve Huggingface-specific configuration details.

        Returns:
            Dict[str, Optional[str]]: Huggingface configuration details.
        """
        return {
            "IMAGES_HUGGINGFACE_BASE_URL": self.base_url,
            "IMAGES_HUGGINGFACE_API_KEY": self.api_key,
            "IMAGES_HUGGINGFACE_ADDITIONAL_HEADERS": json.dumps(self.additional_headers),
        }


# Register the provider
provider_registry.register("huggingface", HuggingfaceProvider)
