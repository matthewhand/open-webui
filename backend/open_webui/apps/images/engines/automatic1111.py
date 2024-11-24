import json
import logging
from typing import List, Dict, Optional

import httpx
from .base import BaseImageEngine
from open_webui.config import (
    AUTOMATIC1111_BASE_URL,
    AUTOMATIC1111_API_AUTH,
    AUTOMATIC1111_CFG_SCALE,
    AUTOMATIC1111_SAMPLER,
    AUTOMATIC1111_SCHEDULER,
    AppConfig,
)

log = logging.getLogger(__name__)

class Automatic1111Engine(BaseImageEngine):
    """
    Engine for AUTOMATIC1111-based image generation.
    """

    def populate_config(self):
        """
        Populate the shared configuration with AUTOMATIC1111-specific details.
        Logs info when required config is available and sets defaults for missing optional fields.
        """
        log.debug("Executing Automatic1111Engine populate_config...")
        config_items = [
            {"key": "AUTOMATIC1111_BASE_URL", "value": AUTOMATIC1111_BASE_URL.value or "http://host.docker.internal:7860", "required": True},
            {"key": "AUTOMATIC1111_API_AUTH", "value": AUTOMATIC1111_API_AUTH.value or "", "required": False},
            {"key": "AUTOMATIC1111_CFG_SCALE", "value": AUTOMATIC1111_CFG_SCALE.value or 7.5, "required": False},
            {"key": "AUTOMATIC1111_SAMPLER", "value": AUTOMATIC1111_SAMPLER.value or "Euler", "required": False},
            {"key": "AUTOMATIC1111_SCHEDULER", "value": AUTOMATIC1111_SCHEDULER.value or "normal", "required": False},
        ]

        missing_fields = []
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
                missing_fields.append(key)
                log.debug(f"Automatic1111Engine: Missing required configuration '{key}'.")

        # Ensure all required and optional fields are set
        self.base_url = getattr(self, "base_url", "")
        self.api_key = getattr(self, "api_key", "")
        self.cfg_scale = getattr(self, "cfg_scale", 7.5)
        self.sampler = getattr(self, "sampler", "Euler")
        self.scheduler = getattr(self, "scheduler", "normal")

        if self.base_url:
            log.info(f"Automatic1111Engine available with base_url: {self.base_url}")
        else:
            log.debug("Automatic1111Engine: Required configuration is missing and engine is not available.")

    def validate_config(self) -> (bool, list):
        """
        Validate the Automatic1111Engine's configuration.

        Returns:
            tuple: (is_valid (bool), missing_fields (list of str))
        """
        missing_configs = []
        if not self.base_url:
            missing_configs.append("AUTOMATIC1111_BASE_URL")

        if missing_configs:
            log.warning(
                f"Automatic1111Engine: Missing required configurations: {', '.join(missing_configs)}."
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
            log.error("Automatic1111Engine is not configured properly.")
            return []

        try:
            width, height = map(int, size.lower().split("x"))
        except ValueError:
            log.error("Invalid size format. Use 'WIDTHxHEIGHT' (e.g., '512x512').")
            return []

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

        log.debug(f"Automatic1111Engine Payload: {payload}")

        headers = self._construct_headers()

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    url=f"{self.base_url}/sdapi/v1/txt2img",
                    headers=headers,
                    json=payload,
                )
            log.debug(f"Automatic1111Engine Response Status: {response.status_code}")
            response.raise_for_status()
            res = response.json()
            log.debug(f"Automatic1111Engine Response: {res}")

            images = []
            for img in res.get("images", []):
                image_filename = self.save_b64_image(img)
                if image_filename:
                    images.append({"url": f"/cache/image/generations/{image_filename}"})
            return images

        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            log.warning(f"Automatic1111Engine Request failed: {e}")
            return []
        except Exception as e:
            log.error(f"Automatic1111Engine Error: {e}")
            return []

    def list_models(self) -> List[Dict[str, str]]:
        """
        List available models for image generation from AUTOMATIC1111's API.

        Returns:
            List[Dict[str, str]]: List of available models.
        """
        if not self.base_url:
            log.error("Automatic1111Engine is not configured properly.")
            return []

        try:
            headers = self._construct_headers()
            with httpx.Client(timeout=10.0) as client:
                response = client.get(
                    url=f"{self.base_url}/sdapi/v1/sd-models",
                    headers=headers,
                )
            response.raise_for_status()
            models = response.json()
            log.debug(f"Automatic1111Engine Models Response: {models}")
            return [
                {"id": model.get("title", "unknown"), "name": model.get("model_name", "unknown")}
                for model in models
            ]
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            log.warning(f"Error listing AUTOMATIC1111 models: {e}")
            return []
        except Exception as e:
            log.error(f"Automatic1111Engine Error during model listing: {e}")
            return []

    def verify_url(self):
        """
        Verify the connectivity of AUTOMATIC1111's API endpoint.
        """
        if not self.base_url:
            log.error("Automatic1111Engine is not configured properly.")
            return {"status": "error", "message": "Automatic1111Engine is not configured."}

        headers = self._construct_headers()

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(
                    url=f"{self.base_url}/sdapi/v1/status",
                    headers=headers,
                )
            response.raise_for_status()
            status = response.json()
            log.info(f"AUTOMATIC1111 API Status: {status}")
            return {"status": "ok", "message": "Automatic1111 API is reachable."}
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            log.warning(f"Could not reach AUTOMATIC1111 API at {self.base_url}: {e}")
            return {"status": "error", "message": f"AUTOMATIC1111 API is unavailable: {e}"}
        except Exception as e:
            log.error(f"Unexpected error verifying AUTOMATIC1111 API: {e}")
            return {"status": "error", "message": "Unexpected error during API verification."}

    def set_model(self, model: str):
        """
        Set the current model for AUTOMATIC1111.

        Args:
            model (str): The model name to set.
        """
        try:
            headers = self._construct_headers()

            # Get current options
            with httpx.Client(timeout=10.0) as client:
                response = client.get(
                    url=f"{self.base_url}/sdapi/v1/options",
                    headers=headers,
                )
            response.raise_for_status()
            options = response.json()

            # Update the model if it's different
            if options.get("sd_model_checkpoint") != model:
                options["sd_model_checkpoint"] = model

                # Set the updated options
                with httpx.Client(timeout=10.0) as client:
                    set_response = client.post(
                        url=f"{self.base_url}/sdapi/v1/options",
                        headers=headers,
                        json=options,
                    )
                set_response.raise_for_status()
                log.info(f"Model set to '{model}' successfully.")
            else:
                log.info(f"Model '{model}' is already set.")
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            log.warning(f"Failed to set model '{model}': {e}")
        except Exception as e:
            log.error(f"Automatic1111Engine Error during set_model: {e}")

    def get_model(self) -> str:
        """
        Get the current model from AUTOMATIC1111.

        Returns:
            str: Currently selected model.
        """
        if not self.base_url:
            log.error("Automatic1111Engine is not configured properly.")
            return ""

        try:
            headers = self._construct_headers()
            with httpx.Client(timeout=10.0) as client:
                response = client.get(
                    url=f"{self.base_url}/sdapi/v1/options",
                    headers=headers,
                )
            response.raise_for_status()
            options = response.json()
            current_model = options.get("sd_model_checkpoint", "")
            log.info(f"Current AUTOMATIC1111 model: '{current_model}'")
            return current_model
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            log.warning(f"Failed to get current model: {e}")
            return ""
        except Exception as e:
            log.error(f"Automatic1111Engine Error during get_model: {e}")
            return ""

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
        Update the shared AppConfig based on form data for AUTOMATIC1111 engine.

        Args:
            form_data (Dict): The form data submitted by the user.
            app_config (AppConfig): The shared configuration object.
        """
        log.debug("Automatic1111Engine updating configuration.")

        # Fallback to AppConfig.ENGINE if "engine" is not in form_data
        engine = form_data.get("engine", "").lower()
        current_engine = getattr(app_config, "ENGINE", "").lower()

        if engine != "automatic1111" and current_engine != "automatic1111":
            log.warning("Automatic1111Engine: Engine not set to 'automatic1111'; skipping config update.")
            return

        # Update model if provided
        if form_data.get("model"):
            self.set_model(form_data["model"])
            log.debug(f"Automatic1111Engine: Model updated to {form_data['model']}")

        # Update image size
        if form_data.get("image_size"):
            app_config.IMAGE_SIZE.value = form_data["image_size"]
            log.debug(f"Automatic1111Engine: IMAGE_SIZE updated to {form_data['image_size']}")

        # Update image steps
        if form_data.get("image_steps") is not None:
            app_config.IMAGE_STEPS.value = form_data["image_steps"]
            log.debug(f"Automatic1111Engine: IMAGE_STEPS updated to {form_data['image_steps']}")

        # Update base URL
        if form_data.get("AUTOMATIC1111_BASE_URL"):
            self.base_url = form_data["AUTOMATIC1111_BASE_URL"]
            log.debug(f"Automatic1111Engine: BASE_URL updated to {form_data['AUTOMATIC1111_BASE_URL']}")

        # Update optional API authentication
        if form_data.get("AUTOMATIC1111_API_AUTH"):
            self.api_key = form_data["AUTOMATIC1111_API_AUTH"]
            log.debug("Automatic1111Engine: API authentication updated.")
