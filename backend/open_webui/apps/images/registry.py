import logging
from typing import Type, Dict, Optional, List
from .base import BaseImageEngine
import open_webui.apps.images.engines

_ = open_webui.apps.images.engines.__all__  # Importing the engines package to trigger their registration

log = logging.getLogger(__name__)

class EngineRegistry:
    """
    A centralized registry for managing image generation engines.

    This class allows registering, retrieving, and listing engines
    that inherit from the BaseImageEngine class.
    """
    _engines: Dict[str, Type[BaseImageEngine]] = {}

    @classmethod
    def register(cls, name: str, engine_class: Type[BaseImageEngine]):
        """
        Register a new engine class.

        Args:
            name (str): The name of the engine (must be unique).
            engine_class (Type[BaseImageEngine]): The engine class to register.

        Raises:
            ValueError: If the engine name is already registered.
            TypeError: If the engine class does not inherit from BaseImageEngine.
        """
        if name in cls._engines:
            log.error(f"Engine '{name}' is already registered.")
            raise ValueError(f"Engine '{name}' is already registered.")
        if not issubclass(engine_class, BaseImageEngine):
            log.error(f"Engine '{name}' must inherit from BaseImageEngine.")
            raise TypeError(f"Engine '{name}' must inherit from BaseImageEngine.")
        cls._engines[name] = engine_class
        log.info(f"Engine '{name}' registered successfully.")

    @classmethod
    def get_engine(cls, name: str) -> Optional[Type[BaseImageEngine]]:
        """
        Retrieve a engine class by name.

        Args:
            name (str): The name of the engine.

        Returns:
            Optional[Type[BaseImageEngine]]: The engine class if found, otherwise None.
        """
        engine = cls._engines.get(name.lower())
        if not engine:
            log.warning(f"Engine '{name}' not found in registry.")
        else:
            log.debug(f"Retrieved engine '{name}': {engine}")
        return engine

    @classmethod
    def list_engines(cls) -> List[str]:
        """
        List all registered engine names.

        Returns:
            List[str]: A list of registered engine names.
        """
        engine_list = list(cls._engines.keys())
        log.debug(f"Registered engines: {engine_list}")
        return engine_list


# Instantiate the registry
engine_registry = EngineRegistry()
