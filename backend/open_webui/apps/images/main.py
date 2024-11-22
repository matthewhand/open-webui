# backend/open_webui/apps/images/main.py

import logging
from typing import Dict, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from open_webui.config import (
    IMAGE_GENERATION_ENGINE,
    ENABLE_IMAGE_GENERATION,
    IMAGE_GENERATION_MODEL,
    IMAGE_SIZE,
    IMAGE_STEPS,
    CORS_ALLOW_ORIGIN,
    AppConfig,
)
from open_webui.env import ENV, SRC_LOG_LEVELS
from open_webui.utils.utils import get_admin_user, get_verified_user
from .providers.registry import provider_registry
from .providers.base import BaseImageProvider

# Initialize logger
log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("IMAGES", logging.INFO))

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

# Dynamically instantiate providers from the registry
PROVIDERS: Dict[str, BaseImageProvider] = {}

for provider_name in provider_registry.list_providers():
    provider_class = provider_registry.get_provider(provider_name)
    if not provider_class:
        log.warning(f"Provider '{provider_name}' not found in registry.")
        continue

    try:
        # Instantiate provider with shared configuration
        provider_instance = provider_class(config=app.state.config)
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
        "enabled": ENABLE_IMAGE_GENERATION.value,
        "engine": IMAGE_GENERATION_ENGINE.value,
        "model": IMAGE_GENERATION_MODEL.value,
        "image_size": IMAGE_SIZE.value,
        "image_steps": IMAGE_STEPS.value,
    }

    # Flatten provider-specific configurations
    flattened_providers = {
        provider_name: provider_instance.get_config()
        for provider_name, provider_instance in PROVIDERS.items()
    }

    # Combine general and provider configurations into a single top-level structure
    return {**general_config, **flattened_providers}

@app.post("/config/update")
async def update_config(form_data: ConfigForm, user=Depends(get_admin_user)):
    """Update application configuration."""
    try:
        # Update ENABLE_IMAGE_GENERATION
        ENABLE_IMAGE_GENERATION.value = form_data.enabled
        ENABLE_IMAGE_GENERATION.save()

        # Update IMAGE_GENERATION_ENGINE
        IMAGE_GENERATION_ENGINE.value = form_data.engine
        IMAGE_GENERATION_ENGINE.save()

        # Update IMAGE_GENERATION_MODEL if provided
        if form_data.model:
            IMAGE_GENERATION_MODEL.value = form_data.model
            IMAGE_GENERATION_MODEL.save()

        # Update IMAGE_SIZE if provided
        if form_data.image_size:
            IMAGE_SIZE.value = form_data.image_size
            IMAGE_SIZE.save()

        # Update IMAGE_STEPS if provided
        if form_data.image_steps:
            IMAGE_STEPS.value = form_data.image_steps
            IMAGE_STEPS.save()

        log.info("Configuration updated via /config/update endpoint.")

    except Exception as e:
        log.exception(f"Failed to update configuration: {e}")
        raise HTTPException(status_code=500, detail="Failed to update configuration.")

    return {"message": "Configuration updated successfully."}

@app.get("/config/url/verify")
async def verify_url(user=Depends(get_admin_user)):
    """
    Verify the connectivity of the configured engine's endpoint.
    """
    engine = IMAGE_GENERATION_ENGINE.value.lower()
    provider: Optional[BaseImageProvider] = PROVIDERS.get(engine)

    if not provider:
        raise HTTPException(status_code=400, detail=f"Engine '{engine}' not supported.")

    try:
        await provider.verify_url()  # Call provider-specific verification
        return {"message": f"Engine '{engine}' verified successfully."}
    except Exception as e:
        log.exception(f"URL verification failed for engine '{engine}': {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_available_models(user=Depends(get_verified_user)):
    """Retrieve models available in the selected engine."""
    engine = IMAGE_GENERATION_ENGINE.value.lower()
    provider: Optional[BaseImageProvider] = PROVIDERS.get(engine)

    if not provider:
        raise HTTPException(status_code=400, detail=f"Engine '{engine}' not supported.")

    try:
        models = await provider.list_models()
        return {"models": models}
    except Exception as e:
        log.exception(f"Failed to retrieve models for engine '{engine}': {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generations")
async def generate_images(form_data: GenerateImageForm, user=Depends(get_verified_user)):
    """Generate images using the selected engine."""
    engine = IMAGE_GENERATION_ENGINE.value.lower()
    provider: Optional[BaseImageProvider] = PROVIDERS.get(engine)

    if not provider:
        raise HTTPException(status_code=400, detail=f"Engine '{engine}' not supported.")

    size = form_data.size or IMAGE_SIZE.value
    try:
        images = await provider.generate_image(
            prompt=form_data.prompt,
            n=form_data.n,
            size=size,
            negative_prompt=form_data.negative_prompt,
        )
        return {"images": images}
    except Exception as e:
        log.exception(f"Image generation failed for engine '{engine}': {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/image/config")
async def get_image_config(user=Depends(get_admin_user)):
    """Retrieve image-specific configuration."""
    return {
        "MODEL": IMAGE_GENERATION_MODEL.value,
        "IMAGE_SIZE": IMAGE_SIZE.value,
        "IMAGE_STEPS": IMAGE_STEPS.value,
    }

def set_image_model(model: str):
    """
    Set the current image model for the selected engine.
    """
    engine = IMAGE_GENERATION_ENGINE.value.lower()
    provider: Optional[BaseImageProvider] = PROVIDERS.get(engine)

    if not provider:
        raise ValueError(f"Engine '{engine}' not supported.")

    # Assuming providers have a `set_model` method
    if hasattr(provider, 'set_model'):
        provider.set_model(model)
    else:
        log.warning(f"Provider '{engine}' does not implement set_model method.")

    # Update the configuration
    IMAGE_GENERATION_MODEL.value = model
    IMAGE_GENERATION_MODEL.save()

def get_image_model():
    """
    Get the current image model for the selected engine.
    """
    engine = IMAGE_GENERATION_ENGINE.value.lower()
    provider: Optional[BaseImageProvider] = PROVIDERS.get(engine)

    if not provider:
        raise ValueError(f"Engine '{engine}' not supported.")

    # Assuming providers have a `get_model` method
    if hasattr(provider, 'get_model'):
        return provider.get_model()
    else:
        log.warning(f"Provider '{engine}' does not implement get_model method.")
        return IMAGE_GENERATION_MODEL.value
