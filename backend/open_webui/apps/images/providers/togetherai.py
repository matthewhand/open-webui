# backend/open_webui/apps/images/providers/togetherai.py

from typing import List, Dict, Optional

from .base import BaseImageProvider

class TogetherAIProvider(BaseImageProvider):
    def generate_image(self, prompt: str, n: int, size: str, negative_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Generate images using TogetherAI's API.

        Args:
            prompt (str): The text prompt for image generation.
            n (int): Number of images to generate.
            size (str): Size of the image (e.g., "512x512").
            negative_prompt (Optional[str]): Negative prompt to exclude certain elements.

        Returns:
            List[Dict[str, str]]: List of image URLs.
        """
        width, height = map(int, size.lower().split('x'))
        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": self.additional_headers.get("steps", 50),
            "n": n,
            "response_format": "b64_json",
        }

        # Log the payload for debugging
        log.debug(f"TogetherAIProvider Payload: {payload}")

        try:
            response = requests.post(
                url=self.base_url,
                headers=self.headers,
                json=payload,
                timeout=(5, 120),
            )
            log.debug(f"TogetherAIProvider Response Status: {response.status_code}")
            response.raise_for_status()
            res = response.json()
            log.debug(f"TogetherAIProvider Response: {res}")

            images = []
            for img in res.get("images", []):
                b64_image = img.get("b64_json")
                if b64_image:
                    image_filename = self.save_b64_image(b64_image)
                    if image_filename:
                        images.append({"url": f"/cache/image/generations/{image_filename}"})
            return images

        except requests.exceptions.RequestException as e:
            log.error(f"TogetherAIProvider Request failed: {e}")
            return [{"url": f"Error: {e}"}]
        except Exception as e:
            log.error(f"TogetherAIProvider Error: {e}")
            return [{"url": f"Error: {e}"}]

