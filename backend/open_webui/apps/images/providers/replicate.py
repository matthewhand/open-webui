# backend/open_webui/apps/images/providers/replicate.py

import json
import logging
from typing import List, Dict, Optional

import httpx

from .base import BaseImageProvider
from .registry import provider_registry
from open_webui.config import app_config  # Import core config

log = logging.getLogger(__name__)

class ReplicateProvider(BaseImageProvider):
    def __init__(self):
        super().__init__(
            base_url=app_config.IMAGES_REPLICATE_BASE_URL,
            api_key=app_config.IMAGES_REPLICATE_API_KEY,
            additional_headers={}  # Add any additional headers if required
        )

    async def generate_image(
        self, prompt: str, n: int, size: str, negative_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        # Implement Replicate-specific image generation logic
        # This is a placeholder implementation
        log.debug(f"ReplicateProvider generating image with prompt: {prompt}, n: {n}, size: {size}")
        try:
            # Example API call to Replicate (modify based on actual API)
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url=self.base_url,
                    headers=self.headers,
                    json={
                        "prompt": prompt,
                        "num_images": n,
                        "size": size,
                        "negative_prompt": negative_prompt or "",
                    },
                    timeout=120.0,
                )
            response.raise_for_status()
            res = response.json()
            log.debug(f"ReplicateProvider Response: {res}")

            images = []
            for img in res.get("images", []):
                b64_image = img.get("b64_json")
                if b64_image:
                    image_filename = self.save_b64_image(b64_image)
                    if image_filename:
                        images.append({"url": f"/cache/image/generations/{image_filename}"})
            return images

        except httpx.RequestError as e:
            log.error(f"ReplicateProvider Request failed: {e}")
            raise Exception(f"ReplicateProvider Request failed: {e}")
        except Exception as e:
            log.error(f"ReplicateProvider Error: {e}")
            raise Exception(f"ReplicateProvider Error: {e}")

    async def list_models(self) -> List[Dict[str, str]]:
        # Implement Replicate-specific model listing logic
        # Placeholder implementation
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url=self.base_url + "/models",  # Adjust endpoint as needed
                    headers=self.headers,
                    timeout=30.0,
                )
            response.raise_for_status()
            models = response.json()
            return [{"id": model["id"], "name": model["name"]} for model in models]
        except Exception as e:
            log.error(f"Error listing Replicate models: {e}")
            return []

# Register the provider
provider_registry.register("replicate", ReplicateProvider)
