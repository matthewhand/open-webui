# backend/open_webui/apps/images/providers/base.py

import base64
import logging
import mimetypes
import os
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional

import aiofiles
import httpx

log = logging.getLogger(__name__)

# Set up a default image cache directory
IMAGE_CACHE_DIR = Path(os.getenv("CACHE_DIR", "./cache")).joinpath("image/generations")
IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class BaseImageProvider(ABC):
    """
    Abstract Base Class for Image Generation Providers.
    Provides common functionality for saving images and managing headers.
    """

    def __init__(self, base_url: str, api_key: str, additional_headers: Optional[Dict[str, str]] = None):
        self.base_url = base_url
        self.api_key = api_key
        self.additional_headers = additional_headers or {}
        self.headers = self._construct_headers()

    def _construct_headers(self) -> Dict[str, str]:
        """
        Construct the headers required for API requests.

        Returns:
            Dict[str, str]: A dictionary of HTTP headers.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.additional_headers)
        return headers

    async def save_b64_image(self, b64_str: str) -> Optional[str]:
        """
        Save a base64-encoded image to the cache directory.

        Args:
            b64_str (str): Base64-encoded image string.

        Returns:
            Optional[str]: Filename of the saved image or None if failed.
        """
        try:
            image_id = str(uuid.uuid4())
            img_data = base64.b64decode(b64_str.split(",")[-1])
            mime_type = self._get_mime_type_from_b64(b64_str)
            image_format = mimetypes.guess_extension(mime_type) or ".png"
            image_filename = f"{image_id}{image_format}"
            file_path = IMAGE_CACHE_DIR / image_filename

            async with aiofiles.open(file_path, "wb") as f:
                await f.write(img_data)

            log.info(f"Image saved as {file_path}")
            return image_filename
        except Exception as e:
            log.exception(f"Error saving base64 image: {e}")
            return None

    async def save_url_image(self, url: str) -> Optional[str]:
        """
        Save an image from a URL to the cache directory.

        Args:
            url (str): URL of the image.

        Returns:
            Optional[str]: Filename of the saved image or None if failed.
        """
        try:
            image_id = str(uuid.uuid4())
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30.0)
                response.raise_for_status()

                if not response.headers.get("content-type", "").startswith("image"):
                    log.error("URL does not point to an image.")
                    return None

                mime_type = response.headers.get("content-type", "image/png")
                image_format = mimetypes.guess_extension(mime_type) or ".png"
                image_filename = f"{image_id}{image_format}"
                file_path = IMAGE_CACHE_DIR / image_filename

                async with aiofiles.open(file_path, "wb") as f:
                    await f.write(response.content)

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
        if "," in b64_str and ";" in b64_str:
            header = b64_str.split(",")[0]
            mime_type = header.split(";")[0].replace("data:", "")
            return mime_type
        return "image/png"

    @abstractmethod
    async def generate_image(
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
    async def list_models(self) -> List[Dict[str, str]]:
        """
        Abstract method to list available models. Must be implemented by subclasses.

        Returns:
            List[Dict[str, str]]: List of available models with 'id' and 'name'.
        """
        pass
