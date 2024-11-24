import asyncio
import base64
import json
import logging
import mimetypes
import re
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

<<<<<<< HEAD
import httpx
=======
import requests
from open_webui.apps.images.utils.comfyui import (
    ComfyUIGenerateImageForm,
    ComfyUIWorkflow,
    comfyui_generate_image,
)
from open_webui.config import (
    AUTOMATIC1111_API_AUTH,
    AUTOMATIC1111_BASE_URL,
    AUTOMATIC1111_CFG_SCALE,
    AUTOMATIC1111_SAMPLER,
    AUTOMATIC1111_SCHEDULER,
    CACHE_DIR,
    COMFYUI_BASE_URL,
    COMFYUI_WORKFLOW,
    COMFYUI_WORKFLOW_NODES,
    CORS_ALLOW_ORIGIN,
    ENABLE_IMAGE_GENERATION,
    IMAGE_GENERATION_ENGINE,
    IMAGE_GENERATION_MODEL,
    IMAGE_SIZE,
    IMAGE_STEPS,
    IMAGES_OPENAI_API_BASE_URL,
    IMAGES_OPENAI_API_KEY,
    AppConfig,
)
from open_webui.constants import ERROR_MESSAGES
from open_webui.env import ENV, SRC_LOG_LEVELS, ENABLE_FORWARD_USER_INFO_HEADERS

>>>>>>> upstream/main
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from open_webui.config import (
    IMAGE_ENABLED_PROVIDERS,
    IMAGE_ENABLED_PROVIDERS_LIST,  # Already parsed as a list
    IMAGE_GENERATION_ENGINE,
    ENABLE_IMAGE_GENERATION,
    IMAGE_GENERATION_MODEL,
    IMAGE_SIZE,
    IMAGE_STEPS,
    CORS_ALLOW_ORIGIN,
    AppConfig,
)
from open_webui.env import ENV, SRC_LOG_LEVELS #, ENABLE_FORWARD_USER_INFO_HEADERS
from open_webui.utils.utils import get_admin_user, get_verified_user

from .providers.registry import provider_registry
from .providers.base import BaseImageProvider

# Initialize logger
logging.basicConfig(
    level=logging.DEBUG,  # Capture all debug logs
    format="%(levelname)s %(asctime)s %(name)s - %(message)s",
)
log = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG)
log.debug("main.py has started execution")  # Top-level confirmation

<<<<<<< HEAD
# FastAPI setup
=======
IMAGE_CACHE_DIR = Path(CACHE_DIR).joinpath("./image/generations/")
IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

>>>>>>> upstream/main
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
    enabled: Optional[bool] = None
    engine: Optional[str] = None
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

# Initialize shared configuration
app.state.config = AppConfig()

app.state.config.ENGINE = IMAGE_GENERATION_ENGINE
app.state.config.ENABLED = ENABLE_IMAGE_GENERATION
app.state.config.MODEL = IMAGE_GENERATION_MODEL
app.state.config.IMAGE_SIZE = IMAGE_SIZE
app.state.config.IMAGE_STEPS = IMAGE_STEPS

# Dynamically instantiate providers from the registry
PROVIDERS: Dict[str, BaseImageProvider] = {}

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

# Enforce default engine on startup
def enforce_default_engine():
    try:
        engine = getattr(app.state.config, "ENGINE", "")
        if isinstance(engine, str):
            engine = engine.lower()
            app.state.config.ENGINE = engine
        else:
            app.state.config.ENGINE = "openai"
            log.warning("ENGINE was not a string. Defaulting to 'openai'.")

        if not engine or engine not in PROVIDERS:
            default_engine = "openai"
            if default_engine in PROVIDERS:
                app.state.config.ENGINE = default_engine
                log.warning(f"Engine was missing or invalid. Defaulting to '{default_engine}'.")
            else:
                log.error(f"Default engine '{default_engine}' is not available in the enabled providers.")
    except Exception as e:
        log.exception(f"Unexpected error during default engine enforcement: {e}")
        app.state.config.ENGINE = "openai"  # Fallback to 'openai'

# Call the function during app initialization, after providers are loaded
enforce_default_engine()

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

# Utility function to fetch and sanitize provider configurations
def get_populated_configs(providers: Dict[str, BaseImageProvider]) -> Dict[str, Dict[str, Any]]:
    """
    Dynamically fetch all configurations from providers,
    substituting missing or None values with empty strings.

    Args:
        providers (Dict[str, BaseImageProvider]): A dictionary of provider instances.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary containing sanitized configurations for each provider.
    """
    populated_config = {}
    for provider_key, provider_instance in providers.items():
        provider_config = provider_instance.get_config()
        # Substitute None with ""
        clean_config = {k: (v if v is not None else "") for k, v in provider_config.items()}
        populated_config[provider_key] = clean_config
    return populated_config

# Routes

@app.get("/config")
def get_config(user=Depends(get_admin_user)):
    """
    Retrieve current configuration, dynamically flattening provider-specific details.
    """
    log.debug("Retrieving current configuration.")
    general_config = {
        "enabled": getattr(app.state.config, "ENABLED", False),
        "engine": getattr(app.state.config, "ENGINE", ""),
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
            app.state.config.ENABLED = data["enabled"]
            log.debug(f"Set ENABLED to {data['enabled']}")

        # Update ENGINE
        if "engine" in data and data["engine"]:
            new_engine = data["engine"].lower()

            if IMAGE_ENABLED_PROVIDERS_LIST and new_engine not in IMAGE_ENABLED_PROVIDERS_LIST:
                raise HTTPException(status_code=400, detail=f"Engine '{new_engine}' is not enabled.")
            if new_engine not in PROVIDERS:
                raise HTTPException(status_code=400, detail=f"Engine '{new_engine}' is not properly configured.")

            app.state.config.ENGINE = new_engine
            log.debug(f"Set ENGINE to '{new_engine}'.")

        # Update active provider configurations
        active_provider = PROVIDERS.get(app.state.config.ENGINE)
        if active_provider:
            provider_config = data.get(app.state.config.ENGINE, {})
            if isinstance(provider_config, dict):
                active_provider.update_config_in_app(provider_config, app.state.config)
                log.debug(f"Updated config for provider '{app.state.config.ENGINE}'.")

        # Populate provider-specific configs
        provider_configs = get_populated_configs(PROVIDERS)

        # Validate the active provider's configuration
        warnings = []
        if active_provider and not active_provider.validate_config():
            warnings.append(f"Provider '{app.state.config.ENGINE}' is not fully configured. Please complete its configuration.")

        # Return a complete response with warnings
        combined_config = {
            "enabled": app.state.config.ENABLED,
            "engine": app.state.config.ENGINE,
            **provider_configs,
        }

        if warnings:
            return JSONResponse(
                status_code=200,
                content={
                    "message": "Configuration updated with warnings.",
                    "config": combined_config,
                    "warnings": warnings,
                },
            )

        log.info("Configuration updated successfully.")
        return combined_config

    except HTTPException as he:
        log.error(f"HTTPException during configuration update: {he.detail}")
        raise he
    except Exception as e:
        log.exception(f"Failed to update configuration: {e}")
        raise HTTPException(status_code=500, detail="Failed to update configuration.")

@app.post("/image/config/update")
def update_image_config(form_data: ImageConfigForm, user=Depends(get_admin_user)):
    """
    Update image-specific configurations such as model, image size, and image steps.
    This route should not affect the engine selection.
    """
    try:
        log.debug(f"Updating image configuration with data: {form_data.dict()}")

        data = form_data.dict()

        # Update MODEL
        if "model" in data and data["model"]:
            log.debug(f"Setting MODEL to {data['model']}")
            set_image_model(data["model"])

        # Update IMAGE_SIZE
        if "image_size" in data and data["image_size"]:
            size_pattern = r"^\d+x\d+$"
            if re.match(size_pattern, data["image_size"]):
                app.state.config.IMAGE_SIZE = data["image_size"]
                log.debug(f"Set IMAGE_SIZE to {data['image_size']}")
            else:
                log.error("Invalid IMAGE_SIZE format received.")
                raise HTTPException(
                    status_code=400,
                    detail="Invalid IMAGE_SIZE format. Use 'WIDTHxHEIGHT' (e.g., 512x512)."
                )

        # Update IMAGE_STEPS
        if "image_steps" in data and data["image_steps"] is not None:
            if isinstance(data["image_steps"], int) and data["image_steps"] >= 0:
                app.state.config.IMAGE_STEPS = data["image_steps"]
                log.debug(f"Set IMAGE_STEPS to {data['image_steps']}")
            else:
                log.error("Invalid IMAGE_STEPS value received.")
                raise HTTPException(
                    status_code=400,
                    detail="Invalid IMAGE_STEPS value. It must be a positive integer."
                )

        # After updating configurations, validate the active provider's configuration
        engine = getattr(app.state.config, "ENGINE", "").lower()
        provider: Optional[BaseImageProvider] = PROVIDERS.get(engine)
        if not provider or not provider.validate_config():
            log.error(f"Validation failed for provider '{engine}'. Please check the configuration.")
            raise HTTPException(
                status_code=400,
                detail=f"Validation failed for provider '{engine}'. Please check the configuration."
            )

        log.info("Image configuration updated successfully via /image/config/update endpoint.")

    except HTTPException as he:
        log.error(f"HTTPException during image configuration update: {he.detail}")
        raise he
    except Exception as e:
        log.exception(f"Failed to update image configuration: {e}")
        raise HTTPException(status_code=500, detail="Failed to update image configuration.")

    # Return the updated image configuration
    image_config = {
        "MODEL": getattr(app.state.config, "MODEL", ""),
        "IMAGE_SIZE": getattr(app.state.config, "IMAGE_SIZE", ""),
        "IMAGE_STEPS": getattr(app.state.config, "IMAGE_STEPS", ""),
    }
    return {"message": "Image configuration updated successfully.", "config": image_config}

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
    except HTTPException as e:
        # Re-raise HTTPExceptions from provider.verify_url()
        raise e
    except Exception as e:
        log.exception(f"URL verification failed for engine '{engine}': {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
def get_models(user=Depends(get_verified_user)):
    """
    Retrieve models available in the selected engine.
    """
    engine = getattr(app.state.config, "ENGINE", "").lower()
    provider: Optional[BaseImageProvider] = PROVIDERS.get(engine)

    log.debug(f"Fetching available models for engine '{engine}'")

    if not provider:
        return {
            "status": "error",
            "message": f"Engine '{engine}' is not supported. Please choose a valid engine.",
            "models": []
        }

    # Validate configuration
    valid, missing_fields = provider.validate_config()
    if not valid:
        return {
            "status": "warning",
            "message": f"Engine '{engine}' is not fully configured. Missing: {', '.join(missing_fields)}",
            "models": []
        }

    try:
        models = provider.list_models()
        return {"status": "ok", "models": models}
    except Exception as e:
        log.exception(f"Failed to retrieve models for engine '{engine}': {e}")
        return {
            "status": "error",
            "message": f"Failed to retrieve models: {e}",
            "models": []
        }

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
<<<<<<< HEAD
        images = provider.generate_image(
            prompt=form_data.prompt,
            n=form_data.n,
            size=size,
            negative_prompt=form_data.negative_prompt,
        )
        log.debug(f"Generated images: {images}")
        return {"images": images}
=======
        if app.state.config.ENGINE == "openai":
            headers = {}
            headers["Authorization"] = f"Bearer {app.state.config.OPENAI_API_KEY}"
            headers["Content-Type"] = "application/json"

            if ENABLE_FORWARD_USER_INFO_HEADERS:
                headers["X-OpenWebUI-User-Name"] = user.name
                headers["X-OpenWebUI-User-Id"] = user.id
                headers["X-OpenWebUI-User-Email"] = user.email
                headers["X-OpenWebUI-User-Role"] = user.role

            data = {
                "model": (
                    app.state.config.MODEL
                    if app.state.config.MODEL != ""
                    else "dall-e-2"
                ),
                "prompt": form_data.prompt,
                "n": form_data.n,
                "size": (
                    form_data.size if form_data.size else app.state.config.IMAGE_SIZE
                ),
                "response_format": "b64_json",
            }

            # Use asyncio.to_thread for the requests.post call
            r = await asyncio.to_thread(
                requests.post,
                url=f"{app.state.config.OPENAI_API_BASE_URL}/images/generations",
                json=data,
                headers=headers,
            )

            r.raise_for_status()
            res = r.json()

            images = []

            for image in res["data"]:
                image_filename = save_b64_image(image["b64_json"])
                images.append({"url": f"/cache/image/generations/{image_filename}"})
                file_body_path = IMAGE_CACHE_DIR.joinpath(f"{image_filename}.json")

                with open(file_body_path, "w") as f:
                    json.dump(data, f)

            return images

        elif app.state.config.ENGINE == "comfyui":
            data = {
                "prompt": form_data.prompt,
                "width": width,
                "height": height,
                "n": form_data.n,
            }

            if app.state.config.IMAGE_STEPS is not None:
                data["steps"] = app.state.config.IMAGE_STEPS

            if form_data.negative_prompt is not None:
                data["negative_prompt"] = form_data.negative_prompt

            form_data = ComfyUIGenerateImageForm(
                **{
                    "workflow": ComfyUIWorkflow(
                        **{
                            "workflow": app.state.config.COMFYUI_WORKFLOW,
                            "nodes": app.state.config.COMFYUI_WORKFLOW_NODES,
                        }
                    ),
                    **data,
                }
            )
            res = await comfyui_generate_image(
                app.state.config.MODEL,
                form_data,
                user.id,
                app.state.config.COMFYUI_BASE_URL,
            )
            log.debug(f"res: {res}")

            images = []

            for image in res["data"]:
                image_filename = save_url_image(image["url"])
                images.append({"url": f"/cache/image/generations/{image_filename}"})
                file_body_path = IMAGE_CACHE_DIR.joinpath(f"{image_filename}.json")

                with open(file_body_path, "w") as f:
                    json.dump(form_data.model_dump(exclude_none=True), f)

            log.debug(f"images: {images}")
            return images
        elif (
            app.state.config.ENGINE == "automatic1111" or app.state.config.ENGINE == ""
        ):
            if form_data.model:
                set_image_model(form_data.model)

            data = {
                "prompt": form_data.prompt,
                "batch_size": form_data.n,
                "width": width,
                "height": height,
            }

            if app.state.config.IMAGE_STEPS is not None:
                data["steps"] = app.state.config.IMAGE_STEPS

            if form_data.negative_prompt is not None:
                data["negative_prompt"] = form_data.negative_prompt

            if app.state.config.AUTOMATIC1111_CFG_SCALE:
                data["cfg_scale"] = app.state.config.AUTOMATIC1111_CFG_SCALE

            if app.state.config.AUTOMATIC1111_SAMPLER:
                data["sampler_name"] = app.state.config.AUTOMATIC1111_SAMPLER

            if app.state.config.AUTOMATIC1111_SCHEDULER:
                data["scheduler"] = app.state.config.AUTOMATIC1111_SCHEDULER

            # Use asyncio.to_thread for the requests.post call
            r = await asyncio.to_thread(
                requests.post,
                url=f"{app.state.config.AUTOMATIC1111_BASE_URL}/sdapi/v1/txt2img",
                json=data,
                headers={"authorization": get_automatic1111_api_auth()},
            )

            res = r.json()
            log.debug(f"res: {res}")

            images = []

            for image in res["images"]:
                image_filename = save_b64_image(image)
                images.append({"url": f"/cache/image/generations/{image_filename}"})
                file_body_path = IMAGE_CACHE_DIR.joinpath(f"{image_filename}.json")

                with open(file_body_path, "w") as f:
                    json.dump({**data, "info": res["info"]}, f)

            return images
>>>>>>> upstream/main
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

        image_config = {
            "MODEL": getattr(app.state.config, "MODEL", ""),
            "IMAGE_SIZE": image_size if image_size else "",
            "IMAGE_STEPS": image_steps if image_steps else "",
        }

        log.debug(f"Image-specific configuration: {image_config}")
        return {"status": "ok", "config": image_config}
    except Exception as e:
        log.error(f"Error retrieving image configuration: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve image configuration.")

# Helper Functions

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

    # Update the configuration directly
    app.state.config.MODEL = model
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
        return getattr(app.state.config, "MODEL", "")
