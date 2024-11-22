import logging
import json
from typing import List, Dict, Optional

import httpx

from .base import BaseImageProvider
from .registry import provider_registry
from open_webui.config import (
    AUTOMATIC1111_API_AUTH,
    AUTOMATIC1111_BASE_URL,
    AUTOMATIC1111_CFG_SCALE,
    AUTOMATIC1111_SAMPLER,
    AUTOMATIC1111_SCHEDULER,
)

log = logging.getLogger(__name__)


class Automatic1111Provider(BaseImageProvider):
    """
    Provider for AUTOMATIC1111-based image generation.
    """

    def __init__(self):
        """
        Initialize the AUTOMATIC1111 provider with its specific configuration.
        """
        super().__init__(
            base_url=str(AUTOMATIC1111_BASE_URL.value),
            api_key=str(AUTOMATIC1111_API_AUTH.value),
            additional_headers={},  # Add any additional headers if required
        )

    async def generate_image(
        self, prompt: str, n: int, size: str, negative_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Generate images using AUTOMATIC1111's API.

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
            "batch_size": n,
            "width": width,
            "height": height,
            "cfg_scale": float(AUTOMATIC1111_CFG_SCALE.value),
            "sampler_name": str(AUTOMATIC1111_SAMPLER.value),
            "scheduler": str(AUTOMATIC1111_SCHEDULER.value),
        }

        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        log.debug(f"Automatic1111Provider Payload: {payload}")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url=f"{self.base_url}/sdapi/v1/txt2img",
                    headers=self.headers,
                    json=payload,
                    timeout=120.0,
                )
            log.debug(f"Automatic1111Provider Response Status: {response.status_code}")
            response.raise_for_status()
            res = response.json()
            log.debug(f"Automatic1111Provider Response: {res}")

            images = []
            for img in res.get("images", []):
                image_filename = await self.save_b64_image(img)
                if image_filename:
                    images.append({"url": f"/cache/image/generations/{image_filename}"})
            return images

        except httpx.RequestError as e:
            log.error(f"Automatic1111Provider Request failed: {e}")
            raise Exception(f"Automatic1111Provider Request failed: {e}")
        except Exception as e:
            log.error(f"Automatic1111Provider Error: {e}")
            raise Exception(f"Automatic1111Provider Error: {e}")

    async def list_models(self) -> List[Dict[str, str]]:
        """
        List available models for image generation from AUTOMATIC1111's API.

        Returns:
            List[Dict[str, str]]: List of available models.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url=f"{self.base_url}/sdapi/v1/sd-models",
                    headers=self.headers,
                    timeout=30.0,
                )
            response.raise_for_status()
            models = response.json()
            return [{"id": model["title"], "name": model["model_name"]} for model in models]
        except Exception as e:
            log.error(f"Error listing AUTOMATIC1111 models: {e}")
            return []

    def get_config(self) -> Dict[str, Optional[str]]:
        """
        Retrieve AUTOMATIC1111-specific configuration details.

        Returns:
            Dict[str, Optional[str]]: Configuration details specific to AUTOMATIC1111.
        """
        return {
            "AUTOMATIC1111_BASE_URL": str(AUTOMATIC1111_BASE_URL.value),
            "AUTOMATIC1111_API_AUTH": str(AUTOMATIC1111_API_AUTH.value),
            "AUTOMATIC1111_CFG_SCALE": str(AUTOMATIC1111_CFG_SCALE.value),
            "AUTOMATIC1111_SAMPLER": str(AUTOMATIC1111_SAMPLER.value),
            "AUTOMATIC1111_SCHEDULER": str(AUTOMATIC1111_SCHEDULER.value),
        }


# Register the provider
provider_registry.register("automatic1111", Automatic1111Provider)
