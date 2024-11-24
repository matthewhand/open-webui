# backend/open_webui/apps/images/providers/replicate.py

import json
import logging
from typing import List, Dict, Optional

import httpx
from .base import BaseImageProvider
from .registry import provider_registry
from open_webui.config import AppConfig

log = logging.getLogger(__name__)


class ReplicateProvider(BaseImageProvider):
    """
    Provider for Replicate API-based image generation.
    """

    def populate_config(self):
        """
        Populate Replicate-specific configuration.
        Logs info when required config is available and skips silently if not configured.
        """
        config_items = [
            {"key": "IMAGES_REPLICATE_BASE_URL", "value": getattr(self.config, "IMAGES_REPLICATE_BASE_URL", "").value, "required": True},
            {"key": "IMAGES_REPLICATE_API_KEY", "value": getattr(self.config, "IMAGES_REPLICATE_API_KEY", "").value, "required": True},
        ]

        for config in config_items:
            key = config["key"]
            value = config["value"]
            required = config["required"]

            if value:
                if key == "IMAGES_REPLICATE_BASE_URL":
                    self.base_url = value
                elif key == "IMAGES_REPLICATE_API_KEY":
                    self.api_key = value
            elif required:
                log.debug("ReplicateProvider: Required configuration is not set.")

        if hasattr(self, 'base_url') and hasattr(self, 'api_key'):
            log.info(f"ReplicateProvider available with base_url: {self.base_url}")
        else:
            log.debug("ReplicateProvider: Required configuration is missing and provider is not available.")

    def validate_config(self) -> (bool, list):
        """
        Validate the ReplicateProvider's configuration.

        Returns:
            tuple: (is_valid (bool), missing_fields (list of str))
        """
        missing_configs = []
        if not self.base_url:
            missing_configs.append("REPLICATE_BASE_URL")
        if not self.api_key:
            log.warning("ReplicateProvider: API key is missing. Limited functionality may be available.")

        if missing_configs:
            log.warning(f"ReplicateProvider: Missing required configurations: {', '.join(missing_configs)}.")
            return False, missing_configs

        return True, []

    def generate_image(
        self, prompt: str, n: int, size: str, negative_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Generate images using Replicate's API.

        Args:
            prompt (str): Text prompt for image generation.
            n (int): Number of images to generate.
            size (str): Dimensions of the image (e.g., "512x512").
            negative_prompt (Optional[str]): Text to avoid in the generated images.

        Returns:
            List[Dict[str, str]]: List of URLs pointing to generated images.
        """
        if not hasattr(self, 'base_url') or not hasattr(self, 'api_key'):
            log.error("ReplicateProvider is not configured properly.")
            raise Exception("ReplicateProvider is not configured.")

        width, height = map(int, size.lower().split("x"))
        payload = {
            "prompt": prompt,
            "num_images": n,
            "size": {"width": width, "height": height},
            "negative_prompt": negative_prompt or "",
        }

        log.debug(f"ReplicateProvider Payload: {json.dumps(payload, indent=2)}")

        try:
            with httpx.Client() as client:
                response = client.post(
                    url=self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=120.0,
                )
            log.debug(f"ReplicateProvider Response Status: {response.status_code}")
            response.raise_for_status()
            res = response.json()
            log.debug(f"ReplicateProvider Response: {json.dumps(res, indent=2)}")

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

    def list_models(self) -> List[Dict[str, str]]:
        """
        List available models from Replicate's API.

        Returns:
            List[Dict[str, str]]: List of available models.
        """
        if not hasattr(self, 'base_url') or not hasattr(self, 'api_key'):
            log.error("ReplicateProvider is not configured properly.")
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
            log.debug(f"ReplicateProvider Models Response: {json.dumps(models, indent=2)}")
            return [{"id": model.get("id", "unknown"), "name": model.get("name", "unknown")} for model in models]
        except Exception as e:
            log.error(f"Error listing Replicate models: {e}")
            return []

    def verify_url(self):
        """
        Verify connectivity to the Replicate API endpoint.
        """
        if not hasattr(self, 'base_url') or not hasattr(self, 'api_key'):
            log.error("ReplicateProvider is not configured properly.")
            raise Exception("ReplicateProvider is not configured.")

        try:
            with httpx.Client() as client:
                response = client.get(
                    url=f"{self.base_url}/v1/health",
                    headers=self.headers,
                    timeout=10.0,
                )
            response.raise_for_status()
            log.info("Replicate API is reachable.")
        except Exception as e:
            log.error(f"Failed to verify Replicate API: {e}")
            raise Exception(f"Failed to verify Replicate API: {e}")

    def get_config(self) -> Dict[str, Optional[str]]:
        """
        Retrieve Replicate-specific configuration details.

        Returns:
            Dict[str, Optional[str]]: Replicate configuration details.
        """
        return {
            "REPLICATE_BASE_URL": self.base_url,
            "REPLICATE_API_KEY": self.api_key,
        }

    def update_config_in_app(self, form_data: Dict, app_config: AppConfig):
        """
        Update the shared AppConfig based on form data for Replicate provider.

        Args:
            form_data (Dict): The form data submitted by the user.
            app_config (AppConfig): The shared configuration object.
        """
        log.debug("ReplicateProvider updating configuration.")
        engine = form_data.get("engine", "").lower()
        if engine != "replicate":
            log.debug("ReplicateProvider: Engine not set to 'replicate'; skipping config update.")
            return

        # Replicate may not have models to set via the provider, but can update image_size and image_steps
        if form_data.get("image_size"):
            app_config.IMAGE_SIZE.value = form_data["image_size"]
            # app_config.IMAGE_SIZE.save()
            log.debug(f"ReplicateProvider: IMAGE_SIZE updated to {form_data['image_size']}")

        if form_data.get("image_steps") is not None:
            app_config.IMAGE_STEPS.value = form_data["image_steps"]
            # app_config.IMAGE_STEPS.save()
            log.debug(f"ReplicateProvider: IMAGE_STEPS updated to {form_data['image_steps']}")
