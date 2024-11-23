import logging
from typing import Dict, Optional, List

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from open_webui.config import (
    ENABLE_IMAGE_GENERATION,
    IMAGE_ENABLED_PROVIDERS,
    IMAGE_ENABLED_PROVIDERS_LIST,  # Already parsed as a list
    IMAGE_GENERATION_ENGINE,
    IMAGE_GENERATION_MODEL,
    IMAGE_SIZE,
    IMAGE_STEPS,
    CORS_ALLOW_ORIGIN,
    AppConfig,
)
from open_webui.env import ENV, SRC_LOG_LEVELS  # TODO: ENABLE_FORWARD_USER_INFO_HEADERS
from open_webui.utils.utils import get_admin_user, get_verified_user

from .providers.registry import provider_registry
from .providers.base import BaseImageProvider

from pathlib import Path
import os
import re

# Initialize logger
logging.basicConfig(
    level=logging.DEBUG,  # Capture all debug logs
    format="%(levelname)s %(asctime)s %(name)s - %(message)s",
)
log = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG)
log.debug("main.py has started execution")  # Top-level confirmation

# FastAPI setup
app = FastAPI(
    docs_url="/docs" if ENV == "dev" else None,
    openapi_url="/openapi.json" if ENV == "dev" else None,
    redoc_url=None,
)

# CORS Middleware configuration
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
    # Provider-specific configurations are allowed as extra fields
    class Config:
        extra = "allow"

class ImageConfigForm(BaseModel):
    model: Optional[str] = None
    image_size: Optional[str] = None
    image_steps: Optional[int] = None

class GenerateImageForm(BaseModel):
    prompt: str
    n: int = 1
    size: Optional[str] = None
    negative_prompt: Optional[str] = None

# Utility function to fetch and sanitize provider configurations
def get_populated_configs(providers: Dict[str, BaseImageProvider]) -> Dict[str, Dict[str, str]]:
    """
    Dynamically fetch all configurations from providers,
    substituting missing or None values with empty strings.
    
    Args:
        providers (Dict[str, BaseImageProvider]): A dictionary of provider instances.
    
    Returns:
        Dict[str, Dict[str, str]]: A dictionary containing sanitized configurations for each provider.
    """
    populated_config = {}
    for provider_key, provider_instance in providers.items():
        provider_config = provider_instance.get_config()
        # Substitute None with ""
        clean_config = {k: (v if v is not None else "") for k, v in provider_config.items()}
        populated_config[provider_key] = clean_config
    return populated_config

# Initialize shared configuration
app.state.config = AppConfig()

app.state.config.ENGINE = IMAGE_GENERATION_ENGINE
app.state.config.ENABLED = ENABLE_IMAGE_GENERATION

app.state.config.MODEL = IMAGE_GENERATION_MODEL

app.state.config.IMAGE_SIZE = IMAGE_SIZE
app.state.config.IMAGE_STEPS = IMAGE_STEPS

# Dynamically instantiate providers from the registry
PROVIDERS: Dict[str, BaseImageProvider] = {}

# Parse enabled providers from configuration (already done in config.py)
# IMAGE_ENABLED_PROVIDERS_LIST is a list of provider names in lowercase

log.debug(f"IMAGE_ENABLED_PROVIDERS raw value: '{IMAGE_ENABLED_PROVIDERS}'")
log.debug(f"Parsed IMAGE_ENABLED_PROVIDERS_LIST: {IMAGE_ENABLED_PROVIDERS_LIST}")

# Load and validate providers
try:
    log.info("Initializing providers...")
    available_providers = provider_registry.list_providers()
    log.debug(f"Available providers from registry: {available_providers}")

    for provider_name in available_providers:
        provider_key = provider_name.lower()

        log.debug(f"Attempting to load provider: {provider_name}")

        # Check if provider is enabled
        if IMAGE_ENABLED_PROVIDERS_LIST and provider_key not in IMAGE_ENABLED_PROVIDERS_LIST:
            log.info(f"Skipping provider '{provider_name}' as it's not in the enabled list.")
            continue

        provider_class = provider_registry.get_provider(provider_name)
        if not provider_class:
            log.warning(f"Provider '{provider_name}' not found in registry.")
            continue

        try:
            # Instantiate provider with shared configuration
            provider_instance = provider_class(config=app.state.config)
            PROVIDERS[provider_key] = provider_instance
            log.info(f"Provider '{provider_name}' loaded successfully.")
        except Exception as e:
            log.error(f"Failed to load provider '{provider_name}': {e}", exc_info=True)
except Exception as e:
    log.critical(f"Critical error during provider initialization: {e}", exc_info=True)
    raise

log.info(f"Provider initialization completed. Loaded providers: {list(PROVIDERS.keys())}")

# Custom Exception Handlers

@app.exception_handler(RequestValidationError)
def validation_exception_handler(request: Request, exc: RequestValidationError):
    log.error(f"Validation error for request {request.method} {request.url}: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "message": "Validation error",
            "errors": exc.errors(),
            "body": exc.body,
        },
    )

# Middleware to log incoming requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    log.debug(f"Incoming request: {request.method} {request.url}")

    try:
        # Debug request body
        body = await request.body()
        if body:
            log.debug(f"Request body: {body.decode('utf-8')}")
        else:
            log.debug("Request body is empty.")
    except Exception as e:
        log.warning(f"Failed to read request body: {e}")

    try:
        response = await call_next(request)
        log.debug(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        log.error(f"Middleware error: {e}")
        raise e

# Routes

@app.get("/config")
def get_config(user=Depends(get_admin_user)):
    """
    Retrieve current configuration, dynamically flattening provider-specific details.
    """
    log.debug("Retrieving current configuration.")
    general_config = {
        "enabled": getattr(app.state.config.ENABLED, "value", False),
        "engine": getattr(app.state.config.ENGINE, "value", ""),
    }

    # Dynamically fetch and sanitize provider configurations
    provider_configs = get_populated_configs(PROVIDERS)

    # Combine general and provider-specific configurations into a single top-level structure
    combined_config = {**general_config, **provider_configs}

    log.debug(f"Combined configuration: {combined_config}")
    return combined_config

@app.post("/config/update")
def update_config(form_data: ConfigForm, user=Depends(get_admin_user)):
    """Update application configuration."""
    try:
        log.debug(f"Updating configuration with data: {form_data.dict()}")

        data = form_data.dict()

        # Update ENABLED
        if "enabled" in data and data["enabled"] is not None:
            if hasattr(app.state.config.ENABLED, "value"):
                app.state.config.ENABLED.value = data["enabled"]
            else:
                app.state.config.ENABLED = data["enabled"]
            # app.state.config.ENABLED.save()  # Uncomment if save is needed
            log.debug(f"Set ENABLED to {data['enabled']}")

        # Update ENGINE if provided
        if "engine" in data and data["engine"]:
            new_engine = data["engine"].lower()

            if hasattr(app.state.config.ENGINE, "value"):
                app.state.config.ENGINE.value = new_engine
            else:
                app.state.config.ENGINE = new_engine
            # app.state.config.ENGINE.save()  # Uncomment if save is needed
            log.debug(f"Set ENGINE to {new_engine}")

            # Verify that the new engine is enabled
            if IMAGE_ENABLED_PROVIDERS_LIST and new_engine not in IMAGE_ENABLED_PROVIDERS_LIST:
                log.error(f"Engine '{new_engine}' is not enabled.")
                raise HTTPException(
                    status_code=400,
                    detail=f"Engine '{new_engine}' is not enabled."
                )

            # Switch active provider based on new engine
            active_provider: Optional[BaseImageProvider] = PROVIDERS.get(new_engine)
            if not active_provider:
                log.error(f"Engine '{new_engine}' is not properly configured.")
                raise HTTPException(
                    status_code=400,
                    detail=f"Engine '{new_engine}' is not properly configured."
                )
            log.debug(f"Active provider set to '{new_engine}'.")

        # Update provider-specific configurations
        for provider_key, provider_instance in PROVIDERS.items():
            provider_config = data.get(provider_key, {})
            if isinstance(provider_config, dict):
                provider_instance.update_config_in_app(provider_config, app.state.config)
                log.debug(f"Provider '{provider_key}' specific configuration updated.")

        # Update general configurations
        if "model" in data and data["model"]:
            log.debug(f"Setting MODEL to {data['model']}")
            set_image_model(data["model"])

        if "image_size" in data and data["image_size"]:
            size_pattern = r"^\d+x\d+$"
            if re.match(size_pattern, data["image_size"]):
                if hasattr(app.state.config.IMAGE_SIZE, "value"):
                    app.state.config.IMAGE_SIZE.value = data["image_size"]
                else:
                    app.state.config.IMAGE_SIZE = data["image_size"]
                # app.state.config.IMAGE_SIZE.save()  # Uncomment if save is needed
                log.debug(f"Set IMAGE_SIZE to {data['image_size']}")
            else:
                log.error("Invalid IMAGE_SIZE format received.")
                raise HTTPException(
                    status_code=400,
                    detail="Invalid IMAGE_SIZE format. Use 'WIDTHxHEIGHT' (e.g., 512x512)."
                )

        if "image_steps" in data and data["image_steps"] is not None:
            if isinstance(data["image_steps"], int) and data["image_steps"] >= 0:
                if hasattr(app.state.config.IMAGE_STEPS, "value"):
                    app.state.config.IMAGE_STEPS.value = data["image_steps"]
                else:
                    app.state.config.IMAGE_STEPS = data["image_steps"]
                # app.state.config.IMAGE_STEPS.save()  # Uncomment if save is needed
                log.debug(f"Set IMAGE_STEPS to {data['image_steps']}")
            else:
                log.error("Invalid IMAGE_STEPS value received.")
                raise HTTPException(
                    status_code=400,
                    detail="Invalid IMAGE_STEPS value. It must be a positive integer."
                )

        log.info("Configuration updated via /config/update endpoint.")

    except HTTPException as he:
        log.error(f"HTTPException during configuration update: {he.detail}")
        raise he
    except Exception as e:
        log.exception(f"Failed to update configuration: {e}")
        raise HTTPException(status_code=500, detail="Failed to update configuration.")

    log.debug("Configuration update completed successfully.")
    return {"message": "Configuration updated successfully."}

@app.get("/config/url/verify")
def verify_url(user=Depends(get_admin_user)):
    """
    Verify the connectivity of the configured engine's endpoint.
    """
    engine = getattr(app.state.config, "ENGINE", "").lower()
    provider: Optional[BaseImageProvider] = PROVIDERS.get(engine)

    log.debug(f"Verifying URL for engine '{engine}'")

    if not provider:
        log.error(f"Engine '{engine}' is not supported.")
        raise HTTPException(status_code=400, detail=f"Engine '{engine}' is not supported.")

    # Check if provider is configured
    if not provider.is_configured():
        log.error(f"Engine '{engine}' is not properly configured.")
        raise HTTPException(status_code=400, detail=f"Engine '{engine}' is not properly configured.")

    try:
        provider.verify_url()  # Call provider-specific verification
        log.info(f"Engine '{engine}' verified successfully.")
        return {"message": f"Engine '{engine}' verified successfully."}
    except Exception as e:
        log.exception(f"URL verification failed for engine '{engine}': {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
def get_available_models(user=Depends(get_verified_user)):
    """
    Retrieve models available in the selected engine.
    """
    engine = getattr(app.state.config, "ENGINE", "").lower()
    provider: Optional[BaseImageProvider] = PROVIDERS.get(engine)

    log.debug(f"Fetching available models for engine '{engine}'")

    if not provider:
        log.error(f"Engine '{engine}' is not supported.")
        raise HTTPException(status_code=400, detail=f"Engine '{engine}' is not supported.")

    # Check if provider is configured
    if not provider.is_configured():
        log.warning(f"Engine '{engine}' is not properly configured.")
        return {"status": "warning", "message": f"Engine '{engine}' is not properly configured."}

    try:
        models = provider.list_models()
        log.debug(f"Available models for engine '{engine}': {models}")
        return {"status": "ok", "models": models}
    except Exception as e:
        log.exception(f"Failed to retrieve models for engine '{engine}': {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generations")
def generate_images(form_data: GenerateImageForm, user=Depends(get_verified_user)):
    """
    Generate images using the selected engine.
    """
    engine = getattr(app.state.config, "ENGINE", "").lower()
    provider: Optional[BaseImageProvider] = PROVIDERS.get(engine)

    log.debug(f"Generating images using engine '{engine}' with data: {form_data.dict()}")

    if not provider:
        log.error(f"Engine '{engine}' is not supported.")
        raise HTTPException(status_code=400, detail=f"Engine '{engine}' is not supported.")

    # Check if provider is configured
    if not provider.is_configured():
        log.error(f"Engine '{engine}' is not properly configured.")
        raise HTTPException(status_code=400, detail=f"Engine '{engine}' is not properly configured.")

    size = form_data.size or getattr(app.state.config, "IMAGE_SIZE", "")
    log.debug(f"Using image size: '{size}'")

    try:
        images = provider.generate_image(
            prompt=form_data.prompt,
            n=form_data.n,
            size=size,
            negative_prompt=form_data.negative_prompt,
        )
        log.debug(f"Generated images: {images}")
        return {"images": images}
    except Exception as e:
        log.exception(f"Image generation failed for engine '{engine}': {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/image/config")
def get_image_config(user=Depends(get_admin_user)):
    """Retrieve image-specific configuration."""
    log.debug("Retrieving image-specific configuration.")
    try:
        model = getattr(app.state.config, "MODEL", "")
        image_size = getattr(app.state.config, "IMAGE_SIZE", "")
        image_steps = getattr(app.state.config, "IMAGE_STEPS", "")

        # If these are configuration objects, access their .value attributes
        if hasattr(model, "value"):
            model = model.value
        if hasattr(image_size, "value"):
            image_size = image_size.value
        if hasattr(image_steps, "value"):
            image_steps = image_steps.value

        image_config = {
            "MODEL": model if model else "",
            "IMAGE_SIZE": image_size if image_size else "",
            "IMAGE_STEPS": image_steps if image_steps else "",
        }

        log.debug(f"Image-specific configuration: {image_config}")
        return {"status": "ok", "config": image_config}
    except Exception as e:
        log.error(f"Error retrieving image configuration: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve image configuration.")

def set_image_model(model: str):
    """
    Set the current image model for the selected engine.
    """
    engine = getattr(app.state.config, "ENGINE", "").lower()
    provider: Optional[BaseImageProvider] = PROVIDERS.get(engine)

    log.debug(f"Setting image model to '{model}' for engine '{engine}'")

    if not provider:
        log.error(f"Engine '{engine}' is not supported.")
        raise ValueError(f"Engine '{engine}' is not supported.")

    # Assuming providers have a `set_model` method
    if hasattr(provider, 'set_model'):
        try:
            provider.set_model(model)
            log.debug(f"Model set to '{model}' successfully for engine '{engine}'")
        except Exception as e:
            log.error(f"Failed to set model for engine '{engine}': {e}")
            raise ValueError(f"Failed to set model for engine '{engine}': {e}")
    else:
        log.warning(f"Provider '{engine}' does not implement set_model method.")
        raise ValueError(f"Provider '{engine}' does not support model updates.")

    # Update the configuration directly using .value
    if hasattr(app.state.config.MODEL, "value"):
        app.state.config.MODEL.value = model
    else:
        app.state.config.MODEL = model
    # app.state.config.MODEL.save()  # Uncomment if save is needed
    log.debug(f"Set MODEL to '{model}'")

def get_image_model():
    """
    Get the current image model for the selected engine.
    """
    engine = getattr(app.state.config, "ENGINE", "").lower()
    provider: Optional[BaseImageProvider] = PROVIDERS.get(engine)

    log.debug(f"Retrieving current image model for engine '{engine}'")

    if not provider:
        log.error(f"Engine '{engine}' is not supported.")
        raise ValueError(f"Engine '{engine}' is not supported.")

    # Assuming providers have a `get_model` method
    if hasattr(provider, 'get_model'):
        try:
            model = provider.get_model()
            log.debug(f"Current model for engine '{engine}': '{model}'")
            return model
        except Exception as e:
            log.error(f"Failed to get model for engine '{engine}': {e}")
            raise ValueError(f"Failed to get model for engine '{engine}': {e}")
    else:
        log.warning(f"Provider '{engine}' does not implement get_model method.")
        model = getattr(app.state.config, "MODEL", "")
        if hasattr(model, "value"):
            return model.value
        return model
