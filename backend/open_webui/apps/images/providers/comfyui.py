import asyncio
import json
import logging
import random
import urllib.parse
from typing import List, Dict, Optional

import websockets
import httpx
from pydantic import BaseModel

from .base import BaseImageProvider
from .registry import provider_registry
from open_webui.config import (
    COMFYUI_BASE_URL,
    COMFYUI_WORKFLOW,
    COMFYUI_WORKFLOW_NODES,
)

log = logging.getLogger(__name__)


class ComfyUINodeInput(BaseModel):
    type: Optional[str] = None
    node_ids: List[str] = []
    key: Optional[str] = "text"
    value: Optional[str] = None


class ComfyUIWorkflowModel(BaseModel):
    workflow: str
    nodes: List[ComfyUINodeInput]


class ComfyUIProvider(BaseImageProvider):
    """
    Provider for ComfyUI-based image generation.
    """

    def __init__(self):
        """
        Initialize the ComfyUI provider with its specific configuration.
        """
        super().__init__(
            base_url=COMFYUI_BASE_URL.value,
            api_key="",  # ComfyUI doesn't require an API key
            additional_headers={},
        )

    async def generate_image(
        self, prompt: str, n: int, size: str, negative_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Generate images using ComfyUI's API.

        Args:
            prompt (str): The text prompt for image generation.
            n (int): Number of images to generate.
            size (str): Dimensions of the image (e.g., "512x512").
            negative_prompt (Optional[str]): Text to exclude from the generated images.

        Returns:
            List[Dict[str, str]]: List of URLs pointing to generated images.
        """
        width, height = map(int, size.lower().split("x"))
        workflow_config = json.loads(COMFYUI_WORKFLOW.value)
        nodes_config = json.loads(COMFYUI_WORKFLOW_NODES.value)

        # Update workflow with inputs
        for node in nodes_config:
            node_id = node.get("node_id")
            node_type = node.get("type")
            if not node_id or not node_type:
                continue

            inputs = workflow_config.get(node_id, {}).get("inputs", {})
            if node_type == "model":
                inputs["model"] = self.additional_headers.get("model", "")
            elif node_type == "prompt":
                inputs["text"] = prompt
            elif node_type == "negative_prompt":
                inputs["text"] = negative_prompt or ""
            elif node_type == "width":
                inputs["width"] = width
            elif node_type == "height":
                inputs["height"] = height
            elif node_type == "steps":
                inputs["steps"] = self.additional_headers.get("steps", 50)
            elif node_type == "seed":
                seed = self.additional_headers.get("seed", random.randint(0, 18446744073709551614))
                inputs["seed"] = seed
            elif node_type == "n":
                inputs["batch_size"] = n

            workflow_config[node_id]["inputs"] = inputs

        workflow_str = json.dumps(workflow_config)
        client_id = str(random.randint(100000, 999999))

        log.debug(f"ComfyUIProvider Payload: workflow={workflow_str}, client_id={client_id}")

        try:
            images = await self._comfyui_generate_image(
                prompt=prompt,
                client_id=client_id,
                base_url=self.base_url,
                workflow=workflow_config,
            )
            return images["data"] if images else []
        except Exception as e:
            log.exception(f"ComfyUIProvider Error during image generation: {e}")
            raise Exception(f"ComfyUIProvider Error: {e}")

    async def list_models(self) -> List[Dict[str, str]]:
        """
        List available models from ComfyUI's API.

        Returns:
            List[Dict[str, str]]: List of available models.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url=f"{self.base_url}/object_info",
                    headers=self.headers,
                    timeout=30.0,
                )
            response.raise_for_status()
            info = response.json()

            workflow = json.loads(COMFYUI_WORKFLOW.value)
            nodes_config = json.loads(COMFYUI_WORKFLOW_NODES.value)

            model_node_id = next(
                (node.get("node_ids")[0] for node in nodes_config if node.get("type") == "model" and node.get("node_ids")),
                None,
            )

            if not model_node_id:
                return []

            node_class_type = workflow.get(model_node_id, {}).get("class_type", "")
            if not node_class_type:
                return []

            model_list_key = next(
                (key for key in info.get(node_class_type, {}).get("input", {}).get("required", {}) if "_name" in key),
                None,
            )

            if not model_list_key:
                return []

            models = info[node_class_type]["input"]["required"][model_list_key][0]
            return [{"id": model, "name": model} for model in models]
        except Exception as e:
            log.error(f"Error listing ComfyUI models: {e}")
            return []

    async def _comfyui_generate_image(
        self, prompt: str, client_id: str, base_url: str, workflow: Dict
    ) -> Optional[Dict[str, List[Dict[str, str]]]]:
        """
        Communicate with ComfyUI via WebSocket to generate images.

        Args:
            prompt (str): The text prompt for image generation.
            client_id (str): Unique client identifier.
            base_url (str): Base URL for the ComfyUI server.
            workflow (Dict): The workflow configuration.

        Returns:
            Optional[Dict[str, List[Dict[str, str]]]]: Generated image URLs.
        """
        ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
        try:
            async with websockets.connect(f"{ws_url}/ws?clientId={client_id}") as ws:
                log.info("WebSocket connection established with ComfyUI.")

                await ws.send(json.dumps({"workflow": workflow}))
                log.info("Workflow sent to ComfyUI.")

                output_images = []
                while True:
                    message = await ws.recv()
                    if isinstance(message, str):
                        data = json.loads(message)
                        if data.get("type") == "executing" and data.get("data", {}).get("prompt_id") == client_id:
                            break

                history = await self._get_history(prompt_id=client_id, base_url=base_url)
                for output in history.get("outputs", {}).values():
                    if "images" in output:
                        for image in output["images"]:
                            url = self.get_image_url(
                                filename=image["filename"],
                                subfolder=image["subfolder"],
                                folder_type=image["type"],
                                base_url=base_url,
                            )
                            output_images.append({"url": url})

                return {"data": output_images}
        except websockets.exceptions.ConnectionClosed as e:
            log.exception(f"WebSocket connection closed: {e}")
            return None
        except Exception as e:
            log.exception(f"Error during ComfyUI image generation: {e}")
            return None

    async def _get_history(self, prompt_id: str, base_url: str) -> Dict:
        """
        Get the history of a prompt from ComfyUI.

        Args:
            prompt_id (str): The prompt identifier.
            base_url (str): Base URL for the ComfyUI server.

        Returns:
            Dict: The history data.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url=f"{base_url}/history/{prompt_id}",
                    headers=self.headers,
                    timeout=30.0,
                )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            log.exception(f"Error retrieving ComfyUI history: {e}")
            return {}

    def get_image_url(self, filename: str, subfolder: str, folder_type: str, base_url: str) -> str:
        """
        Construct the URL to access an image generated by ComfyUI.

        Args:
            filename (str): The image filename.
            subfolder (str): The subfolder where the image is stored.
            folder_type (str): The type of the folder.
            base_url (str): Base URL for the ComfyUI server.

        Returns:
            str: The URL to access the image.
        """
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        return f"{base_url}/view?{url_values}"

    def get_config(self) -> Dict[str, str]:
        """
        Retrieve ComfyUI-specific configuration details.

        Returns:
            Dict[str, str]: ComfyUI configuration details.
        """
        return {
            "COMFYUI_BASE_URL": COMFYUI_BASE_URL.value,
            "COMFYUI_WORKFLOW": COMFYUI_WORKFLOW.value,
            "COMFYUI_WORKFLOW_NODES": COMFYUI_WORKFLOW_NODES.value,
        }


# Register the provider
provider_registry.register("comfyui", ComfyUIProvider)
