# backend/open_webui/apps/images/providers/comfyui.py

import json
import logging
import random
import urllib.parse
from typing import List, Dict, Optional

import httpx
import websocket
from open_webui.config import COMFYUI_BASE_URL, COMFYUI_WORKFLOW, COMFYUI_WORKFLOW_NODES, AppConfig
from .base import BaseImageProvider
from .registry import provider_registry

log = logging.getLogger(__name__)


class ComfyUIProvider(BaseImageProvider):
    """
    Provider for ComfyUI-based image generation.
    """

    def populate_config(self):
        """
        Populate the shared configuration with ComfyUI-specific details.
        Logs info when required config is available and skips silently if not configured.
        """
        config_items = [
            {"key": "COMFYUI_BASE_URL", "value": COMFYUI_BASE_URL.value or "http://host.docker.internal:8188/", "required": True},
            {"key": "COMFYUI_WORKFLOW", "value": COMFYUI_WORKFLOW.value or "{}", "required": False},
            {"key": "COMFYUI_WORKFLOW_NODES", "value": COMFYUI_WORKFLOW_NODES.value or "[]", "required": False},
        ]

        for config in config_items:
            key = config["key"]
            value = config["value"]
            required = config["required"]

            if value:
                if key == "COMFYUI_BASE_URL":
                    self.base_url = value
                elif key == "COMFYUI_WORKFLOW":
                    try:
                        workflow = json.loads(value) if isinstance(value, str) else value
                        self.workflow = json.dumps(workflow, indent=2)  # Escaped string representation
                    except json.JSONDecodeError as e:
                        log.warning(f"Failed to parse {key}: {e}. Defaulting to empty dict.")
                        self.workflow = "{}"
                elif key == "COMFYUI_WORKFLOW_NODES":
                    try:
                        self.workflow_nodes = json.loads(value)
                    except json.JSONDecodeError as e:
                        log.warning(f"Failed to parse {key}: {e}. Defaulting to empty list.")
                        self.workflow_nodes = []
            elif required:
                log.debug(f"ComfyUIProvider: Required configuration '{key}' is not set.")

        # Ensure defaults
        self.base_url = getattr(self, "base_url", "")
        self.workflow = getattr(self, "workflow", "{}")
        self.workflow_nodes = getattr(self, "workflow_nodes", [])

        if self.base_url:
            log.info(f"ComfyUIProvider available with base_url: {self.base_url}")
        else:
            log.debug("ComfyUIProvider: Required configuration is missing and provider is not available.")

    def validate_config(self) -> (bool, list):
        """
        Validate the ComfyUIProvider's configuration.

        Returns:
            tuple: (is_valid (bool), missing_fields (list of str))
        """
        missing_configs = []
        if not self.base_url:
            missing_configs.append("COMFYUI_BASE_URL")
        if not self.workflow:
            log.warning("ComfyUIProvider: Workflow is missing. Limited functionality may be available.")

        if missing_configs:
            log.warning(
                f"ComfyUIProvider: Missing required configurations: {', '.join(missing_configs)}."
            )
            return False, missing_configs

        # Additional validation logic can be added here if needed
        return True, []

    def generate_image(
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
        if not self.base_url:
            log.error("ComfyUIProvider is not configured properly.")
            raise Exception("ComfyUIProvider is not configured.")

        try:
            width, height = map(int, size.lower().split("x"))
        except ValueError:
            log.error("Invalid size format. Use 'WIDTHxHEIGHT' (e.g., '512x512').")
            raise Exception("Invalid size format. Use 'WIDTHxHEIGHT' (e.g., '512x512').")

        updated_workflow = self._update_workflow(
            prompt=prompt, n=n, width=width, height=height, negative_prompt=negative_prompt
        )

        client_id = str(random.randint(100000, 999999))
        log.debug(f"ComfyUIProvider Workflow: {json.dumps(updated_workflow, indent=2)}")

        try:
            images = self._comfyui_generate_image(client_id=client_id, workflow=updated_workflow)
            return images.get("data", [])
        except Exception as e:
            log.exception(f"ComfyUIProvider Error during image generation: {e}")
            raise Exception(f"ComfyUIProvider Error: {e}")

    def list_models(self) -> List[Dict[str, str]]:
        """
        List available models from ComfyUI's API.

        Returns:
            List[Dict[str, str]]: List of available models.
        """
        if not self.base_url:
            log.error("ComfyUIProvider is not configured properly.")
            return []

        headers = self.headers  # Utilize headers from BaseImageProvider

        try:
            with httpx.Client() as client:
                response = client.get(
                    url=f"{self.base_url}/object_info",
                    headers=headers,
                    timeout=30.0,
                )
            response.raise_for_status()
            info = response.json()

            workflow = self.workflow
            model_node_id = self._get_model_node_id()
            if not model_node_id:
                log.warning("No model node found in ComfyUI workflow configuration.")
                return []

            node_class_type = workflow.get(model_node_id, {}).get("class_type", "")
            if not node_class_type:
                log.warning(f"No class_type found for model node '{model_node_id}'.")
                return []

            model_list_key = next(
                (
                    key
                    for key in info.get(node_class_type, {}).get("input", {}).get("required", {})
                    if "_name" in key
                ),
                None,
            )

            if not model_list_key:
                log.warning(
                    f"No model list key found in ComfyUI object_info for class_type '{node_class_type}'."
                )
                return []

            models = info.get(node_class_type, {}).get("input", {}).get("required", {}).get(model_list_key, [])
            return [{"id": model, "name": model} for model in models]
        except Exception as e:
            log.error(f"Error listing ComfyUI models: {e}")
            return []

    def verify_url(self):
        """
        Verify the connectivity of ComfyUI's API endpoint.
        """
        if not self.base_url:
            log.error("ComfyUIProvider is not configured properly.")
            raise Exception("ComfyUIProvider is not configured.")

        headers = self.headers  # Utilize headers from BaseImageProvider

        try:
            with httpx.Client() as client:
                response = client.get(
                    url=f"{self.base_url}/object_info",
                    headers=headers,
                    timeout=10.0,
                )
            response.raise_for_status()
            info = response.json()
            log.info(f"ComfyUI API is reachable. Retrieved object info: {info}")
        except Exception as e:
            log.error(f"Failed to verify ComfyUI API: {e}")
            raise Exception(f"Failed to verify ComfyUI API: {e}")

    def _update_workflow(
        self, prompt: str, n: int, width: int, height: int, negative_prompt: Optional[str]
    ) -> Dict:
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
        workflow = self.workflow
        for node in self.workflow_nodes:
            node_id = node.get("node_ids", [None])[0]
            node_type = node.get("type")
            if not node_id or not node_type:
                continue

            inputs = workflow.get(node_id, {}).get("inputs", {})
            if node_type == "prompt":
                inputs["text"] = prompt
            elif node_type == "negative_prompt":
                inputs["text"] = negative_prompt or ""
            elif node_type == "width":
                inputs["width"] = width
            elif node_type == "height":
                inputs["height"] = height
            elif node_type == "n":
                inputs["batch_size"] = n

            workflow[node_id]["inputs"] = inputs
        return workflow

    def _get_model_node_id(self) -> Optional[str]:
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

    def _comfyui_generate_image(self, client_id: str, workflow: Dict) -> Dict:
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
            ws_full_url = f"{ws_url}/ws?clientId={client_id}"
            log.info(f"Connecting to WebSocket at {ws_full_url}")
            ws = websocket.create_connection(ws_full_url, timeout=30)

            log.info("WebSocket connection established with ComfyUI.")
            ws.send(json.dumps({"workflow": workflow}))
            log.info("Workflow sent to ComfyUI.")

            while True:
                message = ws.recv()
                data = json.loads(message)
                if data.get("type") == "executing" and data.get("data", {}).get("prompt_id") == client_id:
                    break

            ws.close()
            log.info("WebSocket communication completed.")

            return self._get_generated_images(client_id)
        except Exception as e:
            log.error(f"Error during WebSocket communication with ComfyUI: {e}")
            raise Exception(f"Error during WebSocket communication with ComfyUI: {e}")

    def _get_generated_images(self, client_id: str) -> Dict:
        """
        Retrieve generated images from ComfyUI's history.

        Args:
            client_id (str): Unique client identifier.

        Returns:
            Dict: Generated image URLs.
        """
        headers = self.headers  # Utilize headers from BaseImageProvider

        try:
            with httpx.Client() as client:
                response = client.get(
                    f"{self.base_url}/history/{client_id}",
                    headers=headers,
                    timeout=30.0,
                )
            response.raise_for_status()
            history = response.json()
            images = [
                self.get_image_url(**img) for img in history.get("outputs", {}).values()
            ]
            return {"data": images}
        except Exception as e:
            log.error(f"Error retrieving generated images from ComfyUI: {e}")
            raise Exception(f"Error retrieving generated images from ComfyUI: {e}")

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

    def get_config(self) -> Dict[str, str]:
        """
        Retrieve ComfyUI-specific configuration details.

        Returns:
            Dict[str, str]: ComfyUI configuration details.
        """
        return {
            "COMFYUI_BASE_URL": self.base_url,
            "COMFYUI_WORKFLOW": self.workflow,
            "COMFYUI_WORKFLOW_NODES": self.workflow_nodes if self.workflow_nodes else [],
        }

    def set_model(self, model: str):
        """
        Set the current image model for ComfyUI.

        Args:
            model (str): The model name to set.
        """
        try:
            # Update the workflow to set the desired model
            workflow = self.workflow
            model_set = False
            for node in self.workflow_nodes:
                node_id = node.get("node_ids", [None])[0]
                node_type = node.get("type")
                if node_type == "model" and node_id:
                    # Assuming the model node has an input field that specifies the model
                    # The exact key depends on the workflow's structure
                    # Here, we assume it's 'model' or similar
                    inputs = workflow.get(node_id, {}).get("inputs", {})
                    if "model" in inputs:
                        inputs["model"] = model
                    elif "ckpt_name" in inputs:
                        inputs["ckpt_name"] = model
                    else:
                        log.warning(f"Unknown input key for model node '{node_id}'.")
                        continue
                    workflow[node_id]["inputs"] = inputs
                    log.info(f"Model set to '{model}' in node '{node_id}'.")
                    model_set = True
                    break

            if not model_set:
                log.warning("No model node found in the workflow to set the model.")
                raise Exception("No model node found in the workflow to set the model.")

            # Update the internal workflow
            self.workflow = workflow

            # Optionally, save the updated workflow to the configuration if needed
            # Assuming AppConfig has COMFYUI_WORKFLOW and COMFYUI_WORKFLOW_NODES as JSON strings
            if hasattr(self.config, "COMFYUI_WORKFLOW"):
                self.config.COMFYUI_WORKFLOW.value = json.dumps(self.workflow)
                self.config.COMFYUI_WORKFLOW.save()
            if hasattr(self.config, "COMFYUI_WORKFLOW_NODES"):
                self.config.COMFYUI_WORKFLOW_NODES.value = json.dumps(self.workflow_nodes)
                self.config.COMFYUI_WORKFLOW_NODES.save()

            log.info(f"ComfyUIProvider model set to '{model}' successfully.")
        except Exception as e:
            log.error(f"Failed to set model '{model}' in ComfyUIProvider: {e}")
            raise Exception(f"Failed to set model '{model}' in ComfyUIProvider: {e}")

    def get_model(self) -> str:
        """
        Get the current image model from ComfyUI's workflow.

        Returns:
            str: Currently selected model.
        """
        try:
            workflow = self.workflow
            model_node_id = self._get_model_node_id()
            if not model_node_id:
                log.warning("No model node found in the workflow.")
                return ""
            inputs = workflow.get(model_node_id, {}).get("inputs", {})
            # The key to get the model may vary; adjust based on actual workflow
            model = inputs.get("model") or inputs.get("ckpt_name") or ""
            log.info(f"Current ComfyUI model: '{model}'")
            return model
        except Exception as e:
            log.error(f"Failed to get model from ComfyUI workflow: {e}")
            return ""

    def is_configured(self) -> bool:
        return bool(getattr(self, 'base_url', '') and getattr(self, 'workflow', ''))

    def update_config_in_app(self, form_data: Dict, app_config: AppConfig):
        """
        Update the shared AppConfig based on form data for ComfyUI provider.

        Args:
            form_data (Dict): The form data submitted by the user.
            app_config (AppConfig): The shared configuration object.
        """
        log.debug("ComfyUIProvider updating configuration.")
        
        # Fallback to AppConfig.ENGINE if "engine" is not in form_data
        engine = form_data.get("engine", "").lower()
        current_engine = getattr(app_config, "ENGINE", "").lower()

        if engine != "comfyui" and current_engine != "comfyui":
            log.warning("ComfyUIProvider: Engine not set to 'comfyui'; skipping config update.")
            return

        # Update model if provided
        if form_data.get("model"):
            self.set_model(form_data["model"])
            log.debug(f"ComfyUIProvider: Model updated to {form_data['model']}")

        # Update image size
        if form_data.get("image_size"):
            app_config.IMAGE_SIZE.value = form_data["image_size"]
            log.debug(f"ComfyUIProvider: IMAGE_SIZE updated to {form_data['image_size']}")

        # Update image steps
        if form_data.get("image_steps") is not None:
            app_config.IMAGE_STEPS.value = form_data["image_steps"]
            log.debug(f"ComfyUIProvider: IMAGE_STEPS updated to {form_data['image_steps']}")
