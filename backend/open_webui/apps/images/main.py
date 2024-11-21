# backend/open_webui/apps/images/main.py

import logging
from pathlib import Path
from typing import Dict, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from open_webui.config import app_config  # Import core config
from open_webui.constants import ERROR_MESSAGES
from open_webui.env import ENV, SRC_LOG_LEVELS
from open_webui.utils.utils import get_admin_user, get_verified_user
from .providers.registry import provider_registry
from .providers.base import BaseImageProvider

# Initialize logger
log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("IMAGES", logging.INFO))

# Cache directory
IMAGE_CACHE_DIR = Path("./cache/image/generations/")
IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# FastAPI setup
app = FastAPI(
    docs_url="/docs" if ENV == "dev" else None,
    openapi_url="/openapi.json" if ENV == "dev" else None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed
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
app.state.config = app_config  # Use the already initialized AppConfig

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

# Routes
@app.get("/config")
async def get_config(user=Depends(get_admin_user)):
    """Retrieve current configuration."""
    return {
        "enabled": app.state.config.ENABLED,
        "engine": app.state.config.ENGINE,
        "model": app.state.config.MODEL,
        "image_size": app.state.config.IMAGE_SIZE,
        "image_steps": app.state.config.IMAGE_STEPS,
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
