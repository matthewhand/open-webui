# backend/open_webui/apps/images/providers/registry.py

from typing import Type, Dict, Optional, List
from .base import BaseImageProvider

class ProviderRegistry:
    _providers: Dict[str, Type[BaseImageProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_class: Type[BaseImageProvider]):
        cls._providers[name] = provider_class

    @classmethod
    def get_provider(cls, name: str) -> Optional[Type[BaseImageProvider]]:
        return cls._providers.get(name)

    @classmethod
    def list_providers(cls) -> List[str]:
        return list(cls._providers.keys())

# Instantiate the registry
provider_registry = ProviderRegistry()
