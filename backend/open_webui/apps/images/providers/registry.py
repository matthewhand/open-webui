import logging
from typing import Type, Dict, Optional, List
from .base import BaseImageProvider

log = logging.getLogger(__name__)


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

        Raises:
            ValueError: If the provider name is already registered.
            TypeError: If the provider class does not inherit from BaseImageProvider.
        """
        if name in cls._providers:
            log.error(f"Provider '{name}' is already registered.")
            raise ValueError(f"Provider '{name}' is already registered.")
        if not issubclass(provider_class, BaseImageProvider):
            log.error(f"Provider '{name}' must inherit from BaseImageProvider.")
            raise TypeError(f"Provider '{name}' must inherit from BaseImageProvider.")
        cls._providers[name] = provider_class
        log.info(f"Provider '{name}' registered successfully.")

    @classmethod
    def get_provider(cls, name: str) -> Optional[Type[BaseImageProvider]]:
        """
        Retrieve a provider class by name.

        Args:
            name (str): The name of the provider.

        Returns:
            Optional[Type[BaseImageProvider]]: The provider class if found, otherwise None.
        """
        provider = cls._providers.get(name.lower())
        if not provider:
            log.warning(f"Provider '{name}' not found in registry.")
        else:
            log.debug(f"Retrieved provider '{name}': {provider}")
        return provider



    @classmethod
    def list_providers(cls) -> List[str]:
        """
        List all registered provider names.

        Returns:
            List[str]: A list of registered provider names.
        """
        provider_list = list(cls._providers.keys())
        log.debug(f"Registered providers: {provider_list}")
        return provider_list


# Instantiate the registry
provider_registry = ProviderRegistry()
