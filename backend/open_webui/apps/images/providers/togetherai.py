# backend/open_webui/apps/images/providers/togetherai.py

import json
import logging
from typing import List, Dict, Optional

import httpx
from open_webui.config import IMAGES_TOGETHERAI_BASE_URL, IMAGES_TOGETHERAI_API_KEY, AppConfig
from .base import BaseImageProvider
from .registry import provider_registry

log = logging.getLogger(__name__)


class TogetherAIProvider(BaseImageProvider):
    """
    Provider for TogetherAI-based image generation.
    """

    def populate_config(self):
        """
        Populate TogetherAI-specific configuration.
        Logs info when required config is available and skips silently if not configured.
        """
        config_items = [
            {"key": "IMAGES_TOGETHERAI_BASE_URL", "value": getattr(self.config, "IMAGES_TOGETHERAI_BASE_URL", "").value, "required": True},
            {"key": "IMAGES_TOGETHERAI_API_KEY", "value": getattr(self.config, "IMAGES_TOGETHERAI_API_KEY", "").value, "required": True},
        ]

        for config in config_items:
            key = config["key"]
            value = config["value"]
            required = config["required"]

            if value:
                if key == "IMAGES_TOGETHERAI_BASE_URL":
                    self.base_url = value
                elif key == "IMAGES_TOGETHERAI_API_KEY":
                    self.api_key = value
            elif required:
                log.debug("TogetherAIProvider: Required configuration is not set.")

        if hasattr(self, 'base_url') and hasattr(self, 'api_key'):
            log.info(f"TogetherAIProvider available with base_url: {self.base_url}")
        else:
            log.debug("TogetherAIProvider: Required configuration is missing and provider is not available.")

    def generate_image(
        self, prompt: str, n: int, size: str, negative_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Generate images using TogetherAI's API.

        Args:
            prompt (str): Text prompt for image generation.
            n (int): Number of images to generate.
            size (str): Dimensions of the image (e.g., "512x512").
            negative_prompt (Optional[str]): Text to avoid in the generated images.

        Returns:
            List[Dict[str, str]]: List of URLs pointing to generated images.
        """
        if not hasattr(self, 'base_url') or not hasattr(self, 'api_key'):
            log.error("TogetherAIProvider is not configured properly.")
            raise Exception("TogetherAIProvider is not configured.")

        width, height = map(int, size.lower().split("x"))
        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": 50,  # Default steps
            "n": n,
            "negative_prompt": negative_prompt or "",
            "response_format": "b64_json",
        }

        log.debug(f"TogetherAIProvider Payload: {json.dumps(payload, indent=2)}")

        try:
            with httpx.Client() as client:
                response = client.post(
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

    def list_models(self) -> List[Dict[str, str]]:
        """
        List available models for image generation from TogetherAI's API.

        Returns:
            List[Dict[str, str]]: List of available models.
        """
        if not hasattr(self, 'base_url') or not hasattr(self, 'api_key'):
            log.error("TogetherAIProvider is not configured properly.")
            return []

        try:
            with httpx.Client() as client:
                response = client.get(
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

    def verify_url(self):
        """
        Verify the connectivity of the TogetherAI API endpoint.
        """
        if not hasattr(self, 'base_url') or not hasattr(self, 'api_key'):
            log.error("TogetherAIProvider is not configured properly.")
            raise Exception("TogetherAIProvider is not configured.")

        try:
            with httpx.Client() as client:
                response = client.get(url=f"{self.base_url}/health", headers=self.headers, timeout=10.0)
            response.raise_for_status()
            log.info("TogetherAI API is reachable.")
        except Exception as e:
            log.error(f"Failed to verify TogetherAI API: {e}")
            raise Exception(f"Failed to verify TogetherAI API: {e}")

    def get_config(self) -> Dict[str, Optional[str]]:
        """
        Retrieve TogetherAI-specific configuration details.

        Returns:
            Dict[str, Optional[str]]: TogetherAI configuration details.
        """
        return {
            "TOGETHERAI_BASE_URL": self.base_url,
            "TOGETHERAI_API_KEY": self.api_key,
        }

    def update_config_in_app(self, form_data: Dict, app_config: AppConfig):
        """
        Update the shared AppConfig based on form data for TogetherAI provider.

        Args:
            form_data (Dict): The form data submitted by the user.
            app_config (AppConfig): The shared configuration object.
        """
        log.debug("TogetherAIProvider updating configuration.")
        engine = form_data.get("engine", "").lower()
        if engine != "togetherai":
            log.debug("TogetherAIProvider: Engine not set to 'togetherai'; skipping config update.")
            return

        # TogetherAI does not have a model in the ConfigForm, but can update image_size and image_steps
        if form_data.get("image_size"):
            app_config.IMAGE_SIZE.value = form_data["image_size"]
            # app_config.IMAGE_SIZE.save()
            log.debug(f"TogetherAIProvider: IMAGE_SIZE updated to {form_data['image_size']}")

        if form_data.get("image_steps") is not None:
            app_config.IMAGE_STEPS.value = form_data["image_steps"]
            # app_config.IMAGE_STEPS.save()
            log.debug(f"TogetherAIProvider: IMAGE_STEPS updated to {form_data['image_steps']}")
