from .automatic1111 import Automatic1111Provider
from .openai import OpenAIProvider
from .comfyui import ComfyUIProvider
from .registry import provider_registry

# Explicitly register providers
provider_registry.register("automatic1111", Automatic1111Provider)
provider_registry.register("openai", OpenAIProvider)
provider_registry.register("comfyui", ComfyUIProvider)

__all__ = [
    "Automatic1111Provider",
    "OpenAIProvider",
    "ComfyUIProvider",
]
