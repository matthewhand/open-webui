# backend/open_webui/apps/images/main.py

import logging
from pathlib import Path
from typing import Dict, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from open_webui.config import (
    AppConfig,
    IMAGE_GENERATION_ENGINE,
    ENABLE_IMAGE_GENERATION,
    IMAGE_GENERATION_MODEL,
    IMAGE_SIZE,
    IMAGE_STEPS,
    CORS_ALLOW_ORIGIN,
)
from open_webui.constants import ERROR_MESSAGES
from open_webui.env import ENV, SRC_LOG_LEVELS
from open_webui.utils.utils import get_admin_user, get_verified_user
from .providers.registry import provider_registry
from .providers.base import BaseImageProvider

# Initialize logger
log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["IMAGES"])

# FastAPI setup
app = FastAPI(
    docs_url="/docs" if ENV == "dev" else None,
    openapi_url="/openapi.json" if ENV == "dev" else None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGIN,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for configuration updates and image generation
class ConfigForm(BaseModel):
    enabled: bool
    engine: str
    model: Optional[str] = None
    image_size: Optional[str] = None
    image_steps: Optional[int] = None


class GenerateImageForm(BaseModel):
    prompt: str
    n: int = 1
    size: Optional[str] = None
    negative_prompt: Optional[str] = None


# Initialize configuration
app.state.config = AppConfig()

app.state.config.ENGINE = IMAGE_GENERATION_ENGINE
app.state.config.ENABLED = ENABLE_IMAGE_GENERATION
app.state.config.MODEL = IMAGE_GENERATION_MODEL
app.state.config.IMAGE_SIZE = IMAGE_SIZE
app.state.config.IMAGE_STEPS = IMAGE_STEPS

# Dynamically instantiate providers from the registry
PROVIDERS: Dict[str, BaseImageProvider] = {}

for provider_name in provider_registry.list_providers():
    provider_class = provider_registry.get_provider(provider_name)
    if not provider_class:
        log.warning(f"Provider '{provider_name}' not found in registry.")
        continue

    try:
        # Instantiate provider (each provider handles its own config)
        provider_instance = provider_class()
        PROVIDERS[provider_name] = provider_instance
        log.info(f"Provider '{provider_name}' initialized successfully.")
    except Exception as e:
        log.error(f"Failed to initialize provider '{provider_name}': {e}")

@app.get("/config")
async def get_config(user=Depends(get_admin_user)):
    """
    Retrieve current configuration, dynamically flattening provider-specific details.
    """
    # General configuration
    general_config = {
        "enabled": app.state.config.ENABLED.value if hasattr(app.state.config.ENABLED, 'value') else app.state.config.ENABLED,
        "engine": app.state.config.ENGINE.value if hasattr(app.state.config.ENGINE, 'value') else app.state.config.ENGINE,
        #"model": app.state.config.MODEL.value if hasattr(app.state.config.MODEL, 'value') else app.state.config.MODEL,
        #"image_size": app.state.config.IMAGE_SIZE.value if hasattr(app.state.config.IMAGE_SIZE, 'value') else app.state.config.IMAGE_SIZE,
        #"image_steps": app.state.config.IMAGE_STEPS.value if hasattr(app.state.config.IMAGE_STEPS, 'value') else app.state.config.IMAGE_STEPS,
    }

    # Flatten provider-specific configurations
    flattened_providers = {
        provider_name: {
            key: value.value if hasattr(value, "value") else value
            for key, value in provider_instance.get_config().items()
        }
        for provider_name, provider_instance in PROVIDERS.items()
    }

    # Combine general and provider configurations into a single top-level structure
    return {
        **general_config,  # Unpack general configuration
        **flattened_providers,  # Unpack flattened provider configurations
    }

@app.post("/config/update")
async def update_config(form_data: ConfigForm, user=Depends(get_admin_user)):
    """Update application configuration."""
    app.state.config.ENABLED = form_data.enabled
    app.state.config.ENGINE = form_data.engine
    if form_data.model:
        app.state.config.MODEL = form_data.model
    if form_data.image_size:
        app.state.config.IMAGE_SIZE = form_data.image_size
    if form_data.image_steps:
        app.state.config.IMAGE_STEPS = form_data.image_steps

    return {"message": "Configuration updated successfully."}


@app.post("/generations")
async def generate_images(form_data: GenerateImageForm, user=Depends(get_verified_user)):
    """Generate images using the selected engine."""
    engine = app.state.config.ENGINE.lower()
    provider: Optional[BaseImageProvider] = PROVIDERS.get(engine)

    if not provider:
        raise HTTPException(status_code=400, detail=f"Engine '{engine}' not supported.")

    size = form_data.size or app.state.config.IMAGE_SIZE
    try:
        # Delegate image generation to the provider
        images = await provider.generate_image(
            prompt=form_data.prompt,
            n=form_data.n,
            size=size,
            negative_prompt=form_data.negative_prompt,
        )
        return {"images": images}
    except Exception as e:
        log.exception(f"Image generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def get_available_models(user=Depends(get_verified_user)):
    """Retrieve models available in the selected engine."""
    engine = app.state.config.ENGINE.lower()
    provider: Optional[BaseImageProvider] = PROVIDERS.get(engine)

    if not provider:
        raise HTTPException(status_code=400, detail=f"Engine '{engine}' not supported.")

    try:
        # Delegate model listing to the provider
        models = await provider.list_models()
        return {"models": models}
    except Exception as e:
        log.exception(f"Failed to retrieve models: {e}")
        raise HTTPException(status_code=500, detail=str(e))
