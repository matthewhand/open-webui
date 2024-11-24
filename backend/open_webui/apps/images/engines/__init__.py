from .automatic1111 import Automatic1111Provider
from .openai import OpenAIProvider
from .comfyui import ComfyUIProvider
from ..registry import engine_registry

# Explicitly register providers
engine_registry.register("automatic1111", Automatic1111Provider)
engine_registry.register("openai", OpenAIProvider)
engine_registry.register("comfyui", ComfyUIProvider)

__all__ = [
    "Automatic1111Provider",
    "OpenAIProvider",
    "ComfyUIProvider",
]
