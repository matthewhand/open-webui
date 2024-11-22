# backend/open_webui/apps/images/providers/registry.py

from typing import Type, Dict, Optional, List
from .base import BaseImageProvider

class ProviderRegistry:
    """
    A centralized registry for managing image generation providers.

    This class allows registering, retrieving, and listing providers
    that inherit from the BaseImageProvider class.
    """
    _providers: Dict[str, Type[BaseImageProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_class: Type[BaseImageProvider]):
        """
        Register a new provider class.

        Args:
            name (str): The name of the provider (must be unique).
            provider_class (Type[BaseImageProvider]): The provider class to register.
        """
        if name in cls._providers:
            raise ValueError(f"Provider '{name}' is already registered.")
        cls._providers[name] = provider_class

    @classmethod
    def get_provider(cls, name: str) -> Optional[Type[BaseImageProvider]]:
        """
        Retrieve a provider class by name.

        Args:
            name (str): The name of the provider.

        Returns:
            Optional[Type[BaseImageProvider]]: The provider class if found, otherwise None.
        """
        return cls._providers.get(name)

    @classmethod
    def list_providers(cls) -> List[str]:
        """
        List all registered provider names.

        Returns:
            List[str]: A list of registered provider names.
        """
        return list(cls._providers.keys())


# Instantiate the registry
provider_registry = ProviderRegistry()
