from .automatic1111 import Automatic1111Engine
from .openai import OpenAIEngine
from .comfyui import ComfyUIEngine
from .registry import engine_registry

# Explicitly register providers
engine_registry.register("automatic1111", Automatic1111Engine)
engine_registry.register("openai", OpenAIEngine)
engine_registry.register("comfyui", ComfyUIEngine)

__all__ = [
    "Automatic1111Engine",
    "OpenAIEngine",
    "ComfyUIEngine",
]
