# backend/open_webui/apps/images/providers/automatic1111.py

import json
import logging
from typing import List, Dict, Optional

import httpx
from ..base import BaseImageProvider
from ..registry import provider_registry
from open_webui.config import (
    AUTOMATIC1111_BASE_URL,
    AUTOMATIC1111_API_AUTH,
    AUTOMATIC1111_CFG_SCALE,
    AUTOMATIC1111_SAMPLER,
    AUTOMATIC1111_SCHEDULER,
    AppConfig,
)

log = logging.getLogger(__name__)


class Automatic1111Provider(BaseImageProvider):
    """
    Provider for AUTOMATIC1111-based image generation.
    """

    def populate_config(self):
        """
        Populate the shared configuration with AUTOMATIC1111-specific details.
        Logs info when required config is available and sets defaults for missing optional fields.
        """

        log.debug("Executing Automatic1111Provider populate_config...")
        config_items = [
            {"key": "AUTOMATIC1111_BASE_URL", "value": AUTOMATIC1111_BASE_URL.value or "http://host.docker.internal:7860/", "required": True},
            {"key": "AUTOMATIC1111_API_AUTH", "value": AUTOMATIC1111_API_AUTH.value or "", "required": False},
            {"key": "AUTOMATIC1111_CFG_SCALE", "value": AUTOMATIC1111_CFG_SCALE.value or 7.5, "required": False},
            {"key": "AUTOMATIC1111_SAMPLER", "value": AUTOMATIC1111_SAMPLER.value or "Euler", "required": False},
            {"key": "AUTOMATIC1111_SCHEDULER", "value": AUTOMATIC1111_SCHEDULER.value or "normal", "required": False},
        ]

        for config in config_items:
            key = config["key"]
            value = config["value"]
            required = config["required"]

            if value:
                if key == "AUTOMATIC1111_BASE_URL":
                    self.base_url = value
                elif key == "AUTOMATIC1111_API_AUTH":
                    self.api_key = value
                elif key == "AUTOMATIC1111_CFG_SCALE":
                    self.cfg_scale = float(value)
                elif key == "AUTOMATIC1111_SAMPLER":
                    self.sampler = value
                elif key == "AUTOMATIC1111_SCHEDULER":
                    self.scheduler = value
            elif required:
                log.debug(f"Automatic1111Provider: Missing required configuration '{key}'.")

        # Ensure all required and optional fields are set
        self.base_url = getattr(self, "base_url", "")
        self.api_key = getattr(self, "api_key", "")
        self.cfg_scale = getattr(self, "cfg_scale", 7.5)
        self.sampler = getattr(self, "sampler", "Euler")
        self.scheduler = getattr(self, "scheduler", "normal")

        if self.base_url:
            log.info(f"Automatic1111Provider available with base_url: {self.base_url}")
        else:
            log.debug("Automatic1111Provider: Required configuration is missing and provider is not available.")

    def validate_config(self) -> (bool, list):
        """
        Validate the Automatic1111Provider's configuration.

        Returns:
            tuple: (is_valid (bool), missing_fields (list of str))
        """
        missing_configs = []
        if not self.base_url:
            missing_configs.append("AUTOMATIC1111_BASE_URL")

        if missing_configs:
            log.warning(
                f"Automatic1111Provider: Missing required configurations: {', '.join(missing_configs)}."
            )
            return False, missing_configs

        # Additional validation logic can be added here
        return True, []

    def generate_image(
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
        if not self.base_url:
            log.error("Automatic1111Provider is not configured properly.")
            raise Exception("Automatic1111Provider is not configured.")

        try:
            width, height = map(int, size.lower().split("x"))
        except ValueError:
            log.error("Invalid size format. Use 'WIDTHxHEIGHT' (e.g., '512x512').")
            raise Exception("Invalid size format. Use 'WIDTHxHEIGHT' (e.g., '512x512').")

        payload = {
            "prompt": prompt,
            "batch_size": n,
            "width": width,
            "height": height,
            "cfg_scale": self.cfg_scale,
            "sampler_name": self.sampler,
            "scheduler": self.scheduler,
        }

        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        log.debug(f"Automatic1111Provider Payload: {payload}")

        headers = {}
        if self.api_key:
            headers["Authorization"] = self.api_key  # Adjust if the auth scheme differs

        try:
            with httpx.Client() as client:
                response = client.post(
                    url=f"{self.base_url}/sdapi/v1/txt2img",
                    headers=headers,
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

    def list_models(self) -> List[Dict[str, str]]:
        """
        List available models for image generation from AUTOMATIC1111's API.

        Returns:
            List[Dict[str, str]]: List of available models.
        """
        if not self.base_url:
            log.error("Automatic1111Provider is not configured properly.")
            return []

        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = self.api_key

            with httpx.Client() as client:
                response = client.get(
                    url=f"{self.base_url}/sdapi/v1/sd-models",
                    headers=headers,
                    timeout=30.0,
                )
            response.raise_for_status()
            models = response.json()
            log.debug(f"Automatic1111Provider Models Response: {models}")
            return [
                {"id": model.get("title", "unknown"), "name": model.get("model_name", "unknown")}
                for model in models
            ]
        except Exception as e:
            log.error(f"Error listing AUTOMATIC1111 models: {e}")
            return []

    def verify_url(self):
        """
        Verify the connectivity of AUTOMATIC1111's API endpoint.
        """
        if not self.base_url:
            log.error("Automatic1111Provider is not configured properly.")
            raise Exception("Automatic1111Provider is not configured.")

        headers = {}
        if self.api_key:
            headers["Authorization"] = self.api_key

        try:
            with httpx.Client() as client:
                response = client.get(
                    url=f"{self.base_url}/sdapi/v1/status",
                    headers=headers,
                    timeout=10.0,
                )
            response.raise_for_status()
            status = response.json()
            log.info(f"AUTOMATIC1111 API Status: {status}")
        except Exception as e:
            log.error(f"Failed to verify AUTOMATIC1111 API: {e}")
            raise Exception(f"Failed to verify AUTOMATIC1111 API: {e}")

    def set_model(self, model: str):
        """
        Set the current model for AUTOMATIC1111.

        Args:
            model (str): The model name to set.

        Raises:
            Exception: If setting the model fails.
        """
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = self.api_key

            # Get current options
            with httpx.Client() as client:
                response = client.get(
                    url=f"{self.base_url}/sdapi/v1/options",
                    headers=headers,
                    timeout=30.0,
                )
            response.raise_for_status()
            options = response.json()

            # Update the model if it's different
            if options.get("sd_model_checkpoint") != model:
                options["sd_model_checkpoint"] = model

                # Set the updated options
                with httpx.Client() as client:
                    set_response = client.post(
                        url=f"{self.base_url}/sdapi/v1/options",
                        headers=headers,
                        json=options,
                        timeout=30.0,
                    )
                set_response.raise_for_status()
                log.info(f"Model set to '{model}' successfully.")
            else:
                log.info(f"Model '{model}' is already set.")
        except Exception as e:
            log.error(f"Failed to set model '{model}': {e}")
            raise Exception(f"Failed to set model '{model}': {e}")

    def get_model(self) -> str:
        """
        Get the current model from AUTOMATIC1111.

        Returns:
            str: Currently selected model.
        """
        if not self.base_url:
            raise Exception("Automatic1111Provider is not configured.")

        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = self.api_key

            with httpx.Client() as client:
                response = client.get(
                    url=f"{self.base_url}/sdapi/v1/options",
                    headers=headers,
                    timeout=30.0,
                )
            response.raise_for_status()
            options = response.json()
            current_model = options.get("sd_model_checkpoint", "")
            log.info(f"Current AUTOMATIC1111 model: '{current_model}'")
            return current_model
        except Exception as e:
            log.error(f"Failed to get current model: {e}")
            raise Exception(f"Failed to get current model: {e}")

    def get_config(self) -> Dict[str, Optional[str]]:
        """
        Retrieve AUTOMATIC1111-specific configuration details.

        Returns:
            Dict[str, Optional[str]]: Configuration details specific to AUTOMATIC1111.
        """
        return {
            "AUTOMATIC1111_BASE_URL": self.base_url,
            "AUTOMATIC1111_API_AUTH": self.api_key,
            "AUTOMATIC1111_CFG_SCALE": str(self.cfg_scale),  # Ensure consistent string formatting
            "AUTOMATIC1111_SAMPLER": self.sampler,
            "AUTOMATIC1111_SCHEDULER": self.scheduler,
        }

    def is_configured(self) -> bool:
        return bool(getattr(self, 'base_url', ''))

    def update_config_in_app(self, form_data: Dict, app_config: AppConfig):
        """
        Update the shared AppConfig based on form data for AUTOMATIC1111 provider.

        Args:
            form_data (Dict): The form data submitted by the user.
            app_config (AppConfig): The shared configuration object.
        """
        log.debug("Automatic1111Provider updating configuration.")
        
        # Fallback to AppConfig.ENGINE if "engine" is not in form_data
        engine = form_data.get("engine", "").lower()
        current_engine = getattr(app_config, "ENGINE", "").lower()

        if engine != "automatic1111" and current_engine != "automatic1111":
            log.warning("Automatic1111Provider: Engine not set to 'automatic1111'; skipping config update.")
            return

        # Update model if provided
        if form_data.get("model"):
            self.set_model(form_data["model"])
            log.debug(f"Automatic1111Provider: Model updated to {form_data['model']}")

        # Update image size
        if form_data.get("image_size"):
            app_config.IMAGE_SIZE.value = form_data["image_size"]
            log.debug(f"Automatic1111Provider: IMAGE_SIZE updated to {form_data['image_size']}")

        # Update image steps
        if form_data.get("image_steps") is not None:
            app_config.IMAGE_STEPS.value = form_data["image_steps"]
            log.debug(f"Automatic1111Provider: IMAGE_STEPS updated to {form_data['image_steps']}")

        # Update base URL
        if form_data.get("AUTOMATIC1111_BASE_URL"):
            self.base_url = form_data["AUTOMATIC1111_BASE_URL"]
            log.debug(f"Automatic1111Provider: BASE_URL updated to {form_data['AUTOMATIC1111_BASE_URL']}")

        # Update optional API authentication
        if form_data.get("AUTOMATIC1111_API_AUTH"):
            self.api_auth = form_data["AUTOMATIC1111_API_AUTH"]
            log.debug("Automatic1111Provider: API authentication updated.")
