# backend/open_webui/apps/images/providers/automatic1111.py

import logging
from typing import List, Dict, Optional

import httpx

from .base import BaseImageProvider
from .registry import provider_registry
from open_webui.config import (
    AUTOMATIC1111_BASE_URL,
    AUTOMATIC1111_API_AUTH,
    AUTOMATIC1111_CFG_SCALE,
    AUTOMATIC1111_SAMPLER,
    AUTOMATIC1111_SCHEDULER,
)

log = logging.getLogger(__name__)


class Automatic1111Provider(BaseImageProvider):
    """
    Provider for AUTOMATIC1111-based image generation.
    """

    def __init__(self, config: AppConfig):
        """
        Initialize the Automatic1111 provider with its specific configuration.

        Args:
            config (AppConfig): The global application configuration object.
        """
        super().__init__(config=config)
        self.base_url = self.config.AUTOMATIC1111_BASE_URL
        self.api_key = self.config.AUTOMATIC1111_API_AUTH
        self.cfg_scale = self.config.AUTOMATIC1111_CFG_SCALE or 7.5
        self.sampler = self.config.AUTOMATIC1111_SAMPLER or "Euler"
        self.scheduler = self.config.AUTOMATIC1111_SCHEDULER or "normal"

        # Initialize headers, including authorization if API key is provided
        self.headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

        log.debug(f"Automatic1111Provider initialized with base_url: {self.base_url}")

    def populate_config(self):
        """
        Populate the shared configuration with AUTOMATIC1111-specific details.
        """
        # Configuration is managed via PersistentConfig; no need to set it here.
        pass

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
            "cfg_scale": float(self.cfg_scale),
            "sampler_name": self.sampler,
            "scheduler": self.scheduler,
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
                image_filename = self.save_b64_image(img)
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
            log.debug(f"Automatic1111Provider Models Response: {models}")
            return [{"id": model.get("title", "unknown"), "name": model.get("model_name", "unknown")} for model in models]
        except Exception as e:
            log.error(f"Error listing AUTOMATIC1111 models: {e}")
            return []

    async def verify_url(self):
        """
        Verify the connectivity of AUTOMATIC1111's API endpoint.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/sdapi/v1/status", headers=self.headers, timeout=10.0)
            response.raise_for_status()
            status = response.json()
            log.info(f"AUTOMATIC1111 API Status: {status}")
        except Exception as e:
            log.error(f"Failed to verify AUTOMATIC1111 API: {e}")
            raise Exception(f"Failed to verify AUTOMATIC1111 API: {e}")

    def get_config(self) -> Dict[str, Optional[str]]:
        """
        Retrieve AUTOMATIC1111-specific configuration details.

        Returns:
            Dict[str, Optional[str]]: Configuration details specific to AUTOMATIC1111.
        """
        return {
            "AUTOMATIC1111_BASE_URL": self.base_url,
            "AUTOMATIC1111_API_AUTH": self.api_key,
            "AUTOMATIC1111_CFG_SCALE": self.cfg_scale,
            "AUTOMATIC1111_SAMPLER": self.sampler,
            "AUTOMATIC1111_SCHEDULER": self.scheduler,
        }


# Register the provider
provider_registry.register("automatic1111", Automatic1111Provider)
