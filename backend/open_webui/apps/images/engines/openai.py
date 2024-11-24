import logging
import httpx
from typing import List, Dict, Optional

from fastapi import HTTPException
from .base import BaseImageEngine
from open_webui.config import AppConfig, IMAGES_OPENAI_API_BASE_URL, IMAGES_OPENAI_API_KEY

log = logging.getLogger(__name__)

class OpenAIEngine(BaseImageEngine):
    """
    Engine for OpenAI's DALL·E image generation API.
    """

    def populate_config(self):
        """
        Populate OpenAI-specific configuration.
        Logs info when required config is available and skips silently if not configured.
        """
        config_items = [
            {"key": "IMAGES_OPENAI_API_BASE_URL", "value": IMAGES_OPENAI_API_BASE_URL.value or "https://api.openai.com/v1", "required": True},
            {"key": "IMAGES_OPENAI_API_KEY", "value": IMAGES_OPENAI_API_KEY.value or "", "required": True},
        ]

        missing_fields = []
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
                missing_fields.append(key)
                log.warning(f"OpenAIEngine: Missing required configuration '{key}'.")

        # Ensure all required attributes are set
        self.base_url = getattr(self, "base_url", "")
        self.api_key = getattr(self, "api_key", "")

        # Initialize the default model if not already set
        self.current_model = getattr(self, "current_model", "dall-e-3")

        if self.base_url and self.api_key:
            log.info(f"OpenAIEngine available with base_url: {self.base_url}")
        else:
            log.debug("OpenAIEngine: Required configuration is missing and engine is not available.")

    def validate_config(self) -> (bool, list):
        """
        Validate the OpenAIEngine's configuration.

        Returns:
            tuple: (is_valid (bool), missing_fields (list of str))
        """
        missing_configs = []
        if not self.base_url:
            missing_configs.append("IMAGES_OPENAI_API_BASE_URL")
        if not self.api_key:
            log.warning("OpenAIEngine: API key is missing. Limited functionality may be available.")

        if missing_configs:
            log.warning(
                f"OpenAIEngine: Missing required configurations: {', '.join(missing_configs)}."
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
        is_valid, missing_fields = self.validate_config()
        if not is_valid:
            log.error("OpenAIEngine is not configured properly.")
            return []

        try:
            payload = {
                "model": self.get_model(),
                "prompt": prompt,
                "n": n,
                "size": size,
                "response_format": "b64_json",
            }

            if negative_prompt:
                payload["negative_prompt"] = negative_prompt  # OpenAI may support this in the future

            log.debug(f"OpenAIEngine Payload: {payload}")

            headers = self._construct_headers()

            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    url=f"{self.base_url}/images/generations",
                    headers=headers,
                    json=payload,
                )
            log.debug(f"OpenAIEngine Response Status: {response.status_code}")
            response.raise_for_status()
            res = response.json()
            log.debug(f"OpenAIEngine Response: {res}")

            images = []
            for image_data in res.get("data", []):
                b64_image = image_data.get("b64_json")
                if b64_image:
                    # Use the inherited method from BaseImageEngine to save the image
                    image_filename = self.save_b64_image(b64_image)
                    if image_filename:
                        images.append({"url": f"/cache/image/generations/{image_filename}"})
            return images

        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            log.warning(f"OpenAIEngine Request failed: {e}")
            return []
        except Exception as e:
            log.error(f"OpenAIEngine Error: {e}")
            return []

    def list_models(self) -> List[Dict[str, str]]:
        """
        List available models for OpenAI's DALL·E API.

        Returns:
            List[Dict[str, str]]: List of available models.
        """
        # OpenAI has predefined models; no need to fetch from API
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
        if not self.base_url or not self.api_key:
            log.error("OpenAIEngine is not configured properly.")
            return {"status": "error", "message": "OpenAIEngine is not configured."}

        headers = self._construct_headers()

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(
                    url=f"{self.base_url}/models",
                    headers=headers,
                )
            response.raise_for_status()
            models = response.json()
            log.info(f"OpenAI API is reachable. Retrieved models: {models}")
            return {"status": "ok", "message": "OpenAI API is reachable."}
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            log.warning(f"Failed to verify OpenAI API: {e}")
            return {"status": "error", "message": f"Failed to verify OpenAI API: {e}"}
        except Exception as e:
            log.error(f"OpenAIEngine Error during URL verification: {e}")
            return {"status": "error", "message": "Unexpected error during API verification."}

    def set_model(self, model: str):
        """
        Set the current image model for OpenAI's DALL·E API.

        Args:
            model (str): Model ID (e.g., 'dall-e-2' or 'dall-e-3').
        """
        if model not in ["dall-e-2", "dall-e-3"]:
            log.error(f"Model '{model}' is not supported by OpenAIEngine.")
            return

        self.current_model = model
        log.info(f"OpenAIEngine model set to: {self.current_model}")

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
        """
        Retrieve OpenAI-specific configuration details.

        Returns:
            Dict[str, Optional[str]]: OpenAI configuration details.
        """
        try:
            config = {
                "OPENAI_API_BASE_URL": self.base_url,
                "OPENAI_API_KEY": self.api_key,
                "CURRENT_MODEL": self.current_model,
            }
            log.debug(f"OpenAIEngine configuration: {config}")
            return config
        except Exception as e:
            log.error(f"Error retrieving OpenAIEngine config: {e}")
            return {}

    def is_configured(self) -> bool:
        """
        Check if OpenAIEngine is properly configured.
        """
        return bool(getattr(self, 'base_url', "")) and bool(getattr(self, 'api_key', ""))

    def update_config_in_app(self, form_data: Dict, app_config: AppConfig):
        """
        Update the shared AppConfig based on form data for OpenAI engine.

        Args:
            form_data (Dict): The form data submitted by the user.
            app_config (AppConfig): The shared configuration object.
        """
        log.debug("OpenAIEngine updating configuration.")

        # Fallback to AppConfig.ENGINE if "engine" is not in form_data
        engine = form_data.get("engine", "").lower()
        current_engine = getattr(app_config, "ENGINE", "").lower()

        if engine != "openai" and current_engine != "openai":
            log.warning("OpenAIEngine: Engine not set to 'openai'; skipping config update.")
            return

        # Update model if provided
        if form_data.get("model"):
            self.set_model(form_data["model"])
            log.debug(f"OpenAIEngine: Model updated to {form_data['model']}")

        # Update image size
        if form_data.get("image_size"):
            app_config.IMAGE_SIZE.value = form_data["image_size"]
            log.debug(f"OpenAIEngine: IMAGE_SIZE updated to {form_data['image_size']}")

        # Update image steps
        if form_data.get("image_steps") is not None:
            app_config.IMAGE_STEPS.value = form_data["image_steps"]
            log.debug(f"OpenAIEngine: IMAGE_STEPS updated to {form_data['image_steps']}")

        # Additional OpenAI-specific configurations (if any)
        if form_data.get("IMAGES_OPENAI_API_KEY"):
            self.api_key = form_data["IMAGES_OPENAI_API_KEY"]
            log.debug("OpenAIEngine: API key updated.")
        if form_data.get("IMAGES_OPENAI_API_BASE_URL"):
            self.base_url = form_data["IMAGES_OPENAI_API_BASE_URL"]
            log.debug(f"OpenAIEngine: Base URL updated to {form_data['IMAGES_OPENAI_API_BASE_URL']}")
