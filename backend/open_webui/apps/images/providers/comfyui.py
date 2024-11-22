# backend/open_webui/apps/images/providers/comfyui.py

import asyncio
import json
import logging
import random
import urllib.parse
from typing import List, Dict, Optional

import websockets
import httpx

from open_webui.config import (
    COMFYUI_BASE_URL,
    COMFYUI_WORKFLOW,
    COMFYUI_WORKFLOW_NODES,
)
from .base import BaseImageProvider
from .registry import provider_registry

log = logging.getLogger(__name__)


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
            api_key=None,
        )
        self.workflow = json.loads(COMFYUI_WORKFLOW.value)
        self.workflow_nodes = COMFYUI_WORKFLOW_NODES.value

        log.debug(f"ComfyUIProvider initialized with base_url: {self.base_url}")

    def populate_config(self):
        """
        Populate the shared configuration with ComfyUI-specific details.
        """
        COMFYUI_BASE_URL.value = self.base_url
        COMFYUI_WORKFLOW.value = json.dumps(self.workflow)
        COMFYUI_WORKFLOW_NODES.value = json.dumps(self.workflow_nodes)

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
        updated_workflow = self._update_workflow(
            prompt=prompt, n=n, width=width, height=height, negative_prompt=negative_prompt
        )

        client_id = str(random.randint(100000, 999999))
        log.debug(f"ComfyUIProvider Workflow: {json.dumps(updated_workflow, indent=2)}")

        try:
            images = await self._comfyui_generate_image(client_id=client_id, workflow=updated_workflow)
            return images.get("data", [])
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

            model_node_id = self._get_model_node_id()
            if not model_node_id:
                log.warning("No model node found in ComfyUI workflow configuration.")
                return []

            node_class_type = self.workflow.get(model_node_id, {}).get("class_type", "")
            if not node_class_type:
                log.warning(f"No class_type found for model node '{model_node_id}'.")
                return []

            model_list_key = next(
                (key for key in info.get(node_class_type, {}).get("input", {}).get("required", {}) if "_name" in key),
                None,
            )

            if not model_list_key:
                log.warning(f"No model list key found in ComfyUI object_info for class_type '{node_class_type}'.")
                return []

            models = info.get(node_class_type, {}).get("input", {}).get("required", {}).get(model_list_key, [])
            return [{"id": model, "name": model} for model in models]
        except Exception as e:
            log.error(f"Error listing ComfyUI models: {e}")
            return []

    async def verify_url(self):
        """
        Verify the connectivity of ComfyUI's API endpoint.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url=f"{self.base_url}/object_info",
                    headers=self.headers,
                    timeout=10.0,
                )
            response.raise_for_status()
            info = response.json()
            log.info(f"ComfyUI API is reachable. Retrieved object info: {info}")
        except Exception as e:
            log.error(f"Failed to verify ComfyUI API: {e}")
            raise Exception(f"Failed to verify ComfyUI API: {e}")

    def _update_workflow(self, prompt: str, n: int, width: int, height: int, negative_prompt: Optional[str]):
        """
        Update the ComfyUI workflow with user-provided parameters.

        Args:
            prompt (str): The text prompt.
            n (int): Number of images to generate.
            width (int): Image width.
            height (int): Image height.
            negative_prompt (Optional[str]): Text to exclude from the generated images.

        Returns:
            Dict: Updated workflow configuration.
        """
        workflow = self.workflow.copy()
        for node in self.workflow_nodes:
            node_id = node.get("node_ids", [None])[0]
            node_type = node.get("type")
            if not node_id or not node_type:
                continue

            inputs = workflow.get(node_id, {}).get("inputs", {})
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
                inputs["steps"] = int(self.additional_headers.get("steps", 50))
            elif node_type == "seed":
                inputs["seed"] = self.additional_headers.get("seed", random.randint(0, 2**32))
            elif node_type == "n":
                inputs["batch_size"] = n

            workflow[node_id]["inputs"] = inputs
        return workflow

    def _get_model_node_id(self):
        """
        Get the model node ID from the workflow nodes.

        Returns:
            Optional[str]: The model node ID, or None if not found.
        """
        return next(
            (
                node.get("node_ids", [None])[0]
                for node in self.workflow_nodes
                if node.get("type") == "model" and node.get("node_ids")
            ),
            None,
        )

    async def _comfyui_generate_image(self, client_id: str, workflow: Dict) -> Dict:
        """
        Communicate with ComfyUI via WebSocket to generate images.

        Args:
            client_id (str): Unique client identifier.
            workflow (Dict): Workflow configuration.

        Returns:
            Dict: Generated image URLs.
        """
        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        try:
            async with websockets.connect(f"{ws_url}/ws?clientId={client_id}") as ws:
                log.info("WebSocket connection established with ComfyUI.")
                await ws.send(json.dumps({"workflow": workflow}))
                log.info("Workflow sent to ComfyUI.")

                while True:
                    message = await ws.recv()
                    data = json.loads(message)
                    if data.get("type") == "executing" and data.get("data", {}).get("prompt_id") == client_id:
                        break

                return await self._get_generated_images(client_id)
        except Exception as e:
            log.error(f"Error during WebSocket communication with ComfyUI: {e}")
            return {}

    async def _get_generated_images(self, client_id: str) -> Dict:
        """
        Retrieve generated images from ComfyUI's history.

        Args:
            client_id (str): Unique client identifier.

        Returns:
            Dict: Generated image URLs.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/history/{client_id}", headers=self.headers)
            response.raise_for_status()
            history = response.json()
            return {"data": [self.get_image_url(**img) for img in history.get("outputs", {}).values()]}
        except Exception as e:
            log.error(f"Error retrieving generated images from ComfyUI: {e}")
            return {}

    def get_image_url(self, filename: str, subfolder: str, folder_type: str) -> str:
        """
        Construct the URL for a generated image.

        Args:
            filename (str): Filename of the image.
            subfolder (str): Subfolder containing the image.
            folder_type (str): Folder type for the image.

        Returns:
            str: Full URL to the image.
        """
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        return f"{self.base_url}/view?{urllib.parse.urlencode(data)}"


# Register the provider
provider_registry.register("comfyui", ComfyUIProvider)
