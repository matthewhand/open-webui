from .automatic1111 import Automatic1111Provider
from .openai import OpenAIProvider
from .comfyui import ComfyUIProvider
from .huggingface import HuggingfaceProvider
from .replicate import ReplicateProvider
from .togetherai import TogetherAIProvider
from .registry import provider_registry

# Explicitly register providers
provider_registry.register("automatic1111", Automatic1111Provider)
provider_registry.register("openai", OpenAIProvider)
provider_registry.register("comfyui", ComfyUIProvider)
provider_registry.register("huggingface", HuggingfaceProvider)
provider_registry.register("replicate", ReplicateProvider)
provider_registry.register("togetherai", TogetherAIProvider)

__all__ = [
    "Automatic1111Provider",
    "OpenAIProvider",
    "ComfyUIProvider",
    "HuggingfaceProvider",
    "ReplicateProvider",
    "TogetherAIProvider",
]
