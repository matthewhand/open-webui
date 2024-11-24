# backend/open_webui/apps/images/providers/openai.py

import logging
import httpx
from typing import List, Dict, Optional

from fastapi import HTTPException
from .base import BaseImageProvider
from .registry import provider_registry
from open_webui.config import AppConfig, IMAGES_OPENAI_API_BASE_URL, IMAGES_OPENAI_API_KEY

log = logging.getLogger(__name__)

class OpenAIProvider(BaseImageProvider):
    """
    Provider for OpenAI's DALL·E image generation API.
    """

    def populate_config(self):
        """
        Populate OpenAI-specific configuration.
        Logs info when required config is available and skips silently if not configured.
        """
        config_items = [
            {"key": "IMAGES_OPENAI_API_BASE_URL", "value": IMAGES_OPENAI_API_BASE_URL.value or "", "required": True},
            {"key": "IMAGES_OPENAI_API_KEY", "value": IMAGES_OPENAI_API_KEY.value or "", "required": True},
        ]

        for config in config_items:
            key = config["key"]
            value = config["value"]
            required = config["required"]

            if value:
                if key == "IMAGES_OPENAI_API_BASE_URL":
                    self.base_url = value
                elif key == "IMAGES_OPENAI_API_KEY":
                    self.api_key = value
            elif required:
                log.warning(f"OpenAIProvider: Missing required configuration '{key}'.")

        # Ensure all required attributes are set
        self.base_url = getattr(self, "base_url", "")
        self.api_key = getattr(self, "api_key", "")

        # Initialize the default model if not already set
        self.current_model = getattr(self, "current_model", "dall-e-3")

        if self.base_url and self.api_key:
            log.info(f"OpenAIProvider available with base_url: {self.base_url}")
        else:
            log.debug("OpenAIProvider: Required configuration is missing and provider is not available.")

    def validate_config(self) -> (bool, list):
        """
        Validate the OpenAIProvider's configuration.

        Returns:
            tuple: (is_valid (bool), missing_fields (list of str))
        """
        missing_configs = []
        if not self.base_url:
            missing_configs.append("OPENAI_API_BASE_URL")
        if not self.api_key:
            missing_configs.append("OPENAI_API_KEY")

        if missing_configs:
            log.warning(
                f"OpenAIProvider: Missing required configurations: {', '.join(missing_configs)}."
            )
            return False, missing_configs

        # Additional validation logic can be added here
        return True, []

    def generate_image(
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
        if not self.validate_config():
            log.error("OpenAIProvider is not configured properly.")
            raise HTTPException(status_code=500, detail="OpenAIProvider configuration is incomplete.")

        payload = {
            "model": self.get_model(),
            "prompt": prompt,
            "n": n,
            "size": size,
            "response_format": "b64_json",
        }

        if negative_prompt:
            payload["negative_prompt"] = negative_prompt  # OpenAI may support this in the future

        log.debug(f"OpenAIProvider Payload: {payload}")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client() as client:
                response = client.post(
                    url=f"{self.base_url}/images/generations",
                    headers=headers,
                    json=payload,
                    timeout=120.0,
                )
            log.debug(f"OpenAIProvider Response Status: {response.status_code}")
            response.raise_for_status()
            res = response.json()
            log.debug(f"OpenAIProvider Response: {res}")

            images = []
            for image_data in res.get("data", []):
                b64_image = image_data.get("b64_json")
                if b64_image:
                    # Use the inherited method from BaseImageProvider to save the image
                    image_filename = self.save_b64_image(b64_image)
                    if image_filename:
                        images.append({"url": f"/cache/image/generations/{image_filename}"})
            return images

        except httpx.RequestError as e:
            log.error(f"OpenAIProvider Request failed: {e}")
            raise HTTPException(status_code=502, detail=f"OpenAIProvider Request failed: {e}")
        except Exception as e:
            log.error(f"OpenAIProvider Error: {e}")
            raise HTTPException(status_code=500, detail=f"OpenAIProvider Error: {e}")

    def list_models(self) -> List[Dict[str, str]]:
        """
        List available models for OpenAI's DALL·E API.

        Returns:
            List[Dict[str, str]]: List of available models.
        """
        models = [
            {"id": "dall-e-2", "name": "DALL·E 2"},
            {"id": "dall-e-3", "name": "DALL·E 3"},
        ]
        log.debug(f"Available models: {models}")
        return models

    def verify_url(self):
        """
        Verify the connectivity of OpenAI's API endpoint.
        """
        log.debug("Verifying OpenAI API connectivity.")
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            with httpx.Client() as client:
                response = client.get(
                    url=f"{self.base_url}/models",
                    headers=headers,
                    timeout=10.0,
                )
            response.raise_for_status()
            models = response.json()
            log.info(f"OpenAI API is reachable. Retrieved models: {models}")
        except httpx.RequestError as e:
            log.error(f"Failed to verify OpenAI API: {e}")
            raise HTTPException(status_code=502, detail=f"Failed to verify OpenAI API: {e}")
        except Exception as e:
            log.error(f"OpenAIProvider Error during URL verification: {e}")
            raise HTTPException(status_code=500, detail=f"OpenAIProvider Error during URL verification: {e}")

    def set_model(self, model: str):
        """
        Set the current image model for OpenAI's DALL·E API.

        Args:
            model (str): Model ID (e.g., 'dall-e-2' or 'dall-e-3').
        """
        if model not in ["dall-e-2", "dall-e-3"]:
            raise ValueError(f"Model '{model}' is not supported by OpenAIProvider.")
        self.current_model = model
        log.info(f"OpenAIProvider model set to: {self.current_model}")

        # Optionally, save the updated model to the configuration
        if hasattr(self.config, "IMAGE_GENERATION_MODEL"):
            self.config.IMAGE_GENERATION_MODEL.value = model
            self.config.IMAGE_GENERATION_MODEL.save()
            log.debug(f"IMAGE_GENERATION_MODEL updated to '{model}'")

    def get_model(self) -> str:
        """
        Get the current image model for OpenAI's DALL·E API.

        Returns:
            str: Currently selected model.
        """
        return getattr(self, "current_model", "dall-e-2")

    def get_config(self) -> Dict[str, Optional[str]]:
        try:
            config = {
                "OPENAI_API_BASE_URL": self.base_url,
                "OPENAI_API_KEY": self.api_key,
                "CURRENT_MODEL": self.current_model,
            }
            log.debug(f"OpenAIProvider configuration: {config}")
            return config
        except Exception as e:
            log.error(f"Error retrieving OpenAIProvider config: {e}")
            return {}

    def is_configured(self) -> bool:
        """
        Check if OpenAIProvider is properly configured.
        """
        return bool(getattr(self, 'base_url', "")) and bool(getattr(self, 'api_key', ""))

    def update_config_in_app(self, form_data: Dict, app_config: AppConfig):
        """
        Update the shared AppConfig based on form data for OpenAI provider.

        Args:
            form_data (Dict): The form data submitted by the user.
            app_config (AppConfig): The shared configuration object.
        """
        log.debug("OpenAIProvider updating configuration.")
        
        # Fallback to AppConfig.ENGINE if "engine" is not in form_data
        engine = form_data.get("engine", "").lower()
        current_engine = getattr(app_config, "ENGINE", "").lower()

        if engine != "openai" and current_engine != "openai":
            log.warning("OpenAIProvider: Engine not set to 'openai'; skipping config update.")
            return

        # Update model if provided
        if form_data.get("model"):
            self.set_model(form_data["model"])
            log.debug(f"OpenAIProvider: Model updated to {form_data['model']}")

        # Update image size
        if form_data.get("image_size"):
            app_config.IMAGE_SIZE.value = form_data["image_size"]
            log.debug(f"OpenAIProvider: IMAGE_SIZE updated to {form_data['image_size']}")

        # Update image steps
        if form_data.get("image_steps") is not None:
            app_config.IMAGE_STEPS.value = form_data["image_steps"]
            log.debug(f"OpenAIProvider: IMAGE_STEPS updated to {form_data['image_steps']}")

        # Additional OpenAI-specific configurations (if any)
        if form_data.get("api_key"):
            self.api_key = form_data["api_key"]
            log.debug("OpenAIProvider: API key updated.")
        if form_data.get("base_url"):
            self.base_url = form_data["base_url"]
            log.debug(f"OpenAIProvider: Base URL updated to {form_data['base_url']}")
