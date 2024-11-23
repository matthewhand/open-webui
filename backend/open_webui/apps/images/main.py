# backend/open_webui/apps/images/main.py

from typing import Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from open_webui.config import (
    IMAGE_GENERATION_ENGINE,
    ENABLE_IMAGE_GENERATION,
    IMAGE_GENERATION_MODEL,
    IMAGE_SIZE,
    IMAGE_STEPS,
    IMAGE_ENABLED_PROVIDERS_LIST,
    CORS_ALLOW_ORIGIN,
    AppConfig,
)
from open_webui.env import ENV, SRC_LOG_LEVELS
from open_webui.utils.utils import get_admin_user, get_verified_user
from .providers.registry import provider_registry
from .providers.base import BaseImageProvider

import re  # For regex in update_image_config

import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s %(asctime)s %(name)s - %(message)s",
)

log = logging.getLogger(__name__)

# Include this in your main script to ensure proper logging
log.debug("Image aplication starting...")

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
    enabled: Optional[bool] = ENABLE_IMAGE_GENERATION.value
    engine: Optional[str] = None
    model: Optional[str] = None
    image_size: Optional[str] = None
    image_steps: Optional[int] = None
    # Provider-specific configuration fields are populated by the provider code


class ImageConfigForm(BaseModel):
    model: Optional[str] = None
    image_size: Optional[str] = None
    image_steps: Optional[int] = None


class GenerateImageForm(BaseModel):
    prompt: str
    n: int = 1
    size: Optional[str] = None
    negative_prompt: Optional[str] = None


# Initialize shared configuration
app.state.config = AppConfig()

# Assign general configuration values
app.state.config.ENGINE = IMAGE_GENERATION_ENGINE
app.state.config.ENABLED = ENABLE_IMAGE_GENERATION
app.state.config.MODEL = IMAGE_GENERATION_MODEL
app.state.config.IMAGE_SIZE = IMAGE_SIZE
app.state.config.IMAGE_STEPS = IMAGE_STEPS

# Dynamically instantiate providers from the registry
PROVIDERS: Dict[str, BaseImageProvider] = {}

# Load and validate providers
log.info("Initializing providers...")
for provider_name in provider_registry.list_providers():
    provider_key = provider_name.lower()

    # Skip providers not listed in IMAGE_ENABLED_PROVIDERS_LIST if it's set
    if IMAGE_ENABLED_PROVIDERS_LIST and provider_key not in [
        p.lower() for p in IMAGE_ENABLED_PROVIDERS_LIST
    ]:
        log.info(
            f"Provider '{provider_name}' is not listed in IMAGE_ENABLED_PROVIDERS and will be skipped."
        )
        continue

    provider_class = provider_registry.get_provider(provider_name)
    if not provider_class:
        log.warning(f"Provider '{provider_name}' not found in registry.")
        continue

    try:
        # Instantiate provider with shared configuration
        provider_instance = provider_class(config=app.state.config)

        # # Validate the provider configuration
        # if not provider_instance.validate_config():
        #     log.warning(f"Provider '{provider_name}' configuration validation failed.")
        #     # No need to raise an exception here; validation happens on save.
        # else:
        #     log.info(f"Provider '{provider_name}' configuration validated.")

        PROVIDERS[provider_key] = provider_instance
        
    except Exception as e:
        log.error(f"Failed to load provider '{provider_name}': {e}", exc_info=True)

log.info("Provider initialization completed.")

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

# # Middleware for request logging
# @app.middleware("http")
# async def log_requests(request: Request, call_next):
#     log.debug(f"Incoming request: {request.method} {request.url}")
#     try:
#         body = await request.body()
#         if body:
#             log.debug(f"Request body: {body.decode('utf-8')}")
#         else:
#             log.debug("Request body is empty.")
#     except Exception as e:
#         log.debug(f"Failed to read request body: {e}")

#     # Await the response
#     response = await call_next(request)
 
#     log.debug(f"Response status: {response.status_code}")
#     return response

# Routes

@app.get("/config")
def get_config(user=Depends(get_admin_user)):
    """
    Retrieve current configuration without validating providers.
    """
    log.debug("Retrieving current configuration.")
    general_config = {
        "enabled": ENABLE_IMAGE_GENERATION.value,
        "engine": IMAGE_GENERATION_ENGINE.value,
    }

    # Flatten provider-specific configurations
    flattened_providers = {
        provider_name: provider_instance.get_config()
        for provider_name, provider_instance in PROVIDERS.items()
    }

    combined_config = {**general_config, **flattened_providers}
    log.debug(f"Combined configuration: {combined_config}")
    return combined_config


@app.post("/config/update")
def update_config(form_data: ConfigForm, user=Depends(get_admin_user)):
    """Update application configuration."""
    try:
        log.debug(f"Updating configuration with data: {form_data.dict()}")

        # Update ENABLE_IMAGE_GENERATION
        ENABLE_IMAGE_GENERATION.value = form_data.enabled
        ENABLE_IMAGE_GENERATION.save()
        log.debug(f"Set ENABLE_IMAGE_GENERATION to {form_data.enabled}")

        # Update IMAGE_GENERATION_ENGINE if provided
        if form_data.engine:
            new_engine = form_data.engine.lower()

            IMAGE_GENERATION_ENGINE.value = new_engine
            log.debug(f"Setting IMAGE_GENERATION_ENGINE to {new_engine}")
            IMAGE_GENERATION_ENGINE.save()
            log.debug(f"Set IMAGE_GENERATION_ENGINE to {new_engine}")

            ## Validate only the newly selected engine
            #active_provider: Optional[BaseImageProvider] = PROVIDERS.get(new_engine)
            # if active_provider and not active_provider.validate_config():
            #     log.error(f"Engine '{new_engine}' is not properly configured.")
            #     raise HTTPException(
            #         status_code=400,
            #         detail=f"Engine '{new_engine}' is not properly configured."
            #     )

        # Update IMAGE_GENERATION_MODEL if provided
        if form_data.model:
            log.debug(f"Setting IMAGE_GENERATION_MODEL to {form_data.model}")
            set_image_model(form_data.model)

        # Update IMAGE_SIZE if provided
        if form_data.image_size:
            log.debug(f"Setting IMAGE_SIZE to {form_data.image_size}")
            IMAGE_SIZE.value = form_data.image_size
            IMAGE_SIZE.save()

        # Update IMAGE_STEPS if provided
        if form_data.image_steps is not None:
            log.debug(f"Setting IMAGE_STEPS to {form_data.image_steps}")
            IMAGE_STEPS.value = form_data.image_steps
            IMAGE_STEPS.save()

        log.info("Configuration updated via /config/update endpoint.")

    except HTTPException as he:
        log.error(f"HTTPException during configuration update: {he.detail}")
        raise he
    except Exception as e:
        log.exception(f"Failed to update configuration: {e}")
        raise HTTPException(status_code=500, detail="Failed to update configuration.")

    log.debug("Configuration update completed successfully.")
    return {"message": "Configuration updated successfully."}

@app.post("/image/config/update")
def update_image_config(form_data: ImageConfigForm, user=Depends(get_admin_user)):
    """Update image-specific configuration such as model, image size, and steps."""
    try:
        log.debug(f"Updating image configuration with data: {form_data.dict()}")

        # Update the model if provided
        if form_data.model:
            log.debug(f"Setting image model to {form_data.model}")
            set_image_model(form_data.model)

        # Validate and update IMAGE_SIZE if provided
        if form_data.image_size:
            size_pattern = r"^\d+x\d+$"
            if re.match(size_pattern, form_data.image_size):
                log.debug(f"Setting IMAGE_SIZE to {form_data.image_size}")
                IMAGE_SIZE.value = form_data.image_size
                IMAGE_SIZE.save()
            else:
                log.error("Invalid IMAGE_SIZE format received.")
                raise HTTPException(
                    status_code=400,
                    detail="Invalid IMAGE_SIZE format. Use 'WIDTHxHEIGHT' (e.g., 512x512)."
                )

        # Validate and update IMAGE_STEPS if provided
        if form_data.image_steps is not None:
            if form_data.image_steps >= 0:
                log.debug(f"Setting IMAGE_STEPS to {form_data.image_steps}")
                IMAGE_STEPS.value = form_data.image_steps
                IMAGE_STEPS.save()
            else:
                log.error("Invalid IMAGE_STEPS value received.")
                raise HTTPException(
                    status_code=400,
                    detail="Invalid IMAGE_STEPS value. It must be a positive integer."
                )

        log.info("Image-specific configuration updated via /image/config/update endpoint.")

    except HTTPException as he:
        log.error(f"HTTPException during image configuration update: {he.detail}")
        raise he
    except Exception as e:
        log.exception(f"Failed to update image configuration: {e}")
        raise HTTPException(status_code=500, detail="Failed to update image configuration.")

    log.debug("Image configuration update completed successfully.")
    return {
        "MODEL": IMAGE_GENERATION_MODEL.value,
        "IMAGE_SIZE": IMAGE_SIZE.value,
        "IMAGE_STEPS": IMAGE_STEPS.value,
    }


@app.get("/config/url/verify")
def verify_url(user=Depends(get_admin_user)):
    """
    Verify the connectivity of the configured engine's endpoint.
    """
    engine = IMAGE_GENERATION_ENGINE.value.lower()
    provider: Optional[BaseImageProvider] = PROVIDERS.get(engine)

    log.debug(f"Verifying URL for engine '{engine}'")

    if not provider:
        log.error(f"Engine '{engine}' not supported.")
        raise HTTPException(status_code=400, detail=f"Engine '{engine}' is not supported.")

    # Check if provider is configured
    if not hasattr(provider, 'base_url') or (hasattr(provider, 'api_key') and not provider.api_key):
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
    No validation performed here; assumes engine is already configured.
    """
    engine = IMAGE_GENERATION_ENGINE.value.lower()
    provider: Optional[BaseImageProvider] = PROVIDERS.get(engine)

    log.debug(f"Fetching available models for engine '{engine}'")

    if not provider:
        log.error(f"Engine '{engine}' not supported.")
        raise HTTPException(status_code=400, detail=f"Engine '{engine}' is not supported.")

    try:
        models = provider.list_models()
        log.debug(f"Available models for engine '{engine}': {models}")
        return {"models": models}
    except Exception as e:
        log.exception(f"Failed to retrieve models for engine '{engine}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generations")
def generate_images(form_data: GenerateImageForm, user=Depends(get_verified_user)):
    """Generate images using the selected engine."""
    engine = IMAGE_GENERATION_ENGINE.value.lower()
    provider: Optional[BaseImageProvider] = PROVIDERS.get(engine)

    log.debug(f"Generating images using engine '{engine}' with data: {form_data.dict()}")

    if not provider:
        log.error(f"Engine '{engine}' not supported.")
        raise HTTPException(status_code=400, detail=f"Engine '{engine}' is not supported.")

    # Check if provider is configured
    if not hasattr(provider, 'base_url') or (hasattr(provider, 'api_key') and not provider.api_key):
        log.error(f"Engine '{engine}' is not properly configured.")
        raise HTTPException(status_code=400, detail=f"Engine '{engine}' is not properly configured.")

    size = form_data.size or IMAGE_SIZE.value
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

    log.debug(f"Setting image model to '{model}' for engine '{engine}'")

    if not provider:
        log.error(f"Engine '{engine}' not supported.")
        raise ValueError(f"Engine '{engine}' not supported.")

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

    # Update the configuration
    IMAGE_GENERATION_MODEL.value = model
    IMAGE_GENERATION_MODEL.save()
    log.debug(f"IMAGE_GENERATION_MODEL updated to '{model}'")


def get_image_model():
    """
    Get the current image model for the selected engine.
    """
    engine = IMAGE_GENERATION_ENGINE.value.lower()
    provider: Optional[BaseImageProvider] = PROVIDERS.get(engine)

    log.debug(f"Retrieving current image model for engine '{engine}'")

    if not provider:
        log.error(f"Engine '{engine}' not supported.")
        raise ValueError(f"Engine '{engine}' not supported.")

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
        return IMAGE_GENERATION_MODEL.value
