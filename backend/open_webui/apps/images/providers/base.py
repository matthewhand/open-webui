import base64
import logging
import mimetypes
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional

import httpx
from open_webui.config import AppConfig, CACHE_DIR

log = logging.getLogger(__name__)

# Calculate and create the cache directory
IMAGE_CACHE_DIR = Path(CACHE_DIR).joinpath("./image/generations/")
IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class BaseImageProvider(ABC):
    """
    Abstract Base Class for Image Generation Providers.
    Provides common functionality for saving images and managing headers.
    """

    def __init__(self, config: AppConfig):
        """
        Initialize the provider with shared configurations.

        Args:
            config (AppConfig): Shared configuration object.
        """
        log.debug("Initializing BaseImageProvider...")
        self.config = config
        # Ensure subclass implements populate_config
        self.populate_config()
        self.headers = self._construct_headers()
        log.debug(f"BaseImageProvider initialized with headers: {self.headers}")

    def _construct_headers(self) -> Dict[str, str]:
        """
        Construct the headers required for API requests.

        Returns:
            Dict[str, str]: A dictionary of HTTP headers.
        """
        log.debug("Constructing headers for API requests...")
        headers = {}
        if hasattr(self, 'api_key') and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers["Content-Type"] = "application/json"
        if hasattr(self, 'additional_headers'):
            headers.update(self.additional_headers)
        log.debug(f"Constructed headers: {headers}")
        return headers

    @abstractmethod
    def populate_config(self):
        """
        Populate the shared configuration with provider-specific details.
        This method must be implemented by subclasses to define their configuration logic.
        """
        pass

    def save_b64_image(self, b64_str: str) -> Optional[str]:
        """
        Save a base64-encoded image to the cache directory.

        Args:
            b64_str (str): Base64-encoded image string.

        Returns:
            Optional[str]: Filename of the saved image or None if failed.
        """
        log.debug("Saving base64-encoded image...")
        log.debug(f"Base64 string length: {len(b64_str)}")
        try:
            image_id = str(uuid.uuid4())
            # Handle data URI scheme if present
            if "," in b64_str:
                b64_str = b64_str.split(",")[-1]
            img_data = base64.b64decode(b64_str)
            mime_type = self._get_mime_type_from_b64(b64_str)
            image_format = mimetypes.guess_extension(mime_type) or ".png"
            image_filename = f"{image_id}{image_format}"
            file_path = IMAGE_CACHE_DIR / image_filename

            with open(file_path, "wb") as f:
                f.write(img_data)

            log.info(f"Image saved as {file_path}")
            return image_filename
        except Exception as e:
            log.exception(f"Error saving base64 image: {e}")
            return None

    def save_url_image(self, url: str) -> Optional[str]:
        """
        Save an image from a URL to the cache directory.

        Args:
            url (str): URL of the image.

        Returns:
            Optional[str]: Filename of the saved image or None if failed.
        """
        log.debug(f"Saving image from URL: {url}")
        try:
            image_id = str(uuid.uuid4())
            with httpx.Client() as client:
                response = client.get(url, timeout=30.0)
                response.raise_for_status()

                if not response.headers.get("content-type", "").startswith("image"):
                    log.error("URL does not point to an image.")
                    return None

                mime_type = response.headers.get("content-type", "image/png")
                image_format = mimetypes.guess_extension(mime_type) or ".png"
                image_filename = f"{image_id}{image_format}"
                file_path = IMAGE_CACHE_DIR / image_filename

                with open(file_path, "wb") as f:
                    f.write(response.content)

            log.info(f"Image downloaded and saved as {file_path}")
            return image_filename
        except Exception as e:
            log.exception(f"Error saving image from URL: {e}")
            return None

    def _get_mime_type_from_b64(self, b64_str: str) -> str:
        """
        Extract the MIME type from a base64-encoded string.

        Args:
            b64_str (str): Base64-encoded string containing MIME type information.

        Returns:
            str: MIME type of the image.
        """
        log.debug("Extracting MIME type from base64 string...")
        if "," in b64_str and ";" in b64_str:
            header = b64_str.split(",")[0]
            mime_type = header.split(";")[0].replace("data:", "")
            log.debug(f"Extracted MIME type: {mime_type}")
            return mime_type
        log.debug("Defaulting MIME type to 'image/png'")
        return "image/png"

    def get_config(self) -> Dict[str, Optional[str]]:
        """
        Return provider-specific configuration details.

        Returns:
            Dict[str, Optional[str]]: Provider-specific configuration details.
        """
        config = {
            "base_url": getattr(self, 'base_url', None),
            "api_key": getattr(self, 'api_key', None),
            "additional_headers": getattr(self, 'additional_headers', {}),
        }
        log.debug(f"Returning provider configuration: {config}")
        return config

    @abstractmethod
    def generate_image(
        self, prompt: str, n: int, size: str, negative_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Abstract method to generate images. Must be implemented by subclasses.

        Args:
            prompt (str): The text prompt for image generation.
            n (int): Number of images to generate.
            size (str): Size of the image (e.g., "512x512").
            negative_prompt (Optional[str]): Negative prompt to exclude certain elements.

        Returns:
            List[Dict[str, str]]: List of image URLs.
        """
        pass

    @abstractmethod
    def list_models(self) -> List[Dict[str, str]]:
        """
        Abstract method to list available models. Must be implemented by subclasses.

        Returns:
            List[Dict[str, str]]: List of available models with 'id' and 'name'.
        """
        pass

    @abstractmethod
    def verify_url(self):
        """
        Abstract method to verify the connectivity of the provider's API endpoint.
        Must be implemented by subclasses.
        """
        pass
