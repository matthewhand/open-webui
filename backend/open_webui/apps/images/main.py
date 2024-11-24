import logging
import re
from typing import Optional, Dict, Any

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
    CORS_ALLOW_ORIGIN,
    AppConfig,
)
from open_webui.env import ENV, SRC_LOG_LEVELS
from open_webui.utils.utils import get_admin_user, get_verified_user

from .engines.registry import engine_registry
from .engines.base import BaseImageEngine

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["IMAGES"])

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
    enabled: Optional[bool] = None
    engine: Optional[str] = None
    # Engine-specific configurations are allowed as extra fields
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

# Dynamically instantiate engines from the registry
ENGINES: Dict[str, BaseImageEngine] = {}

def construct_headers(engine_instance: BaseImageEngine) -> Dict[str, str]:
    """
    Helper method to construct headers for API requests.
    """
    return engine_instance._construct_headers()

# Load and validate engines
try:
    log.info("Initializing engines...")
    available_engines = engine_registry.list_engines()
    log.debug(f"Available engines from registry: {available_engines}")

    for engine_name in available_engines:
        engine_key = engine_name.lower()

        log.debug(f"Attempting to load engine: {engine_name}")

        engine_class = engine_registry.get_engine(engine_name)
        if not engine_class:
            log.warning(f"Engine '{engine_name}' not found in registry.")
            continue

        try:
            # Instantiate engine with shared configuration
            engine_instance = engine_class(config=app.state.config)
            ENGINES[engine_key] = engine_instance
            log.info(f"Engine '{engine_name}' loaded successfully.")
        except Exception as e:
            log.error(f"Failed to load engine '{engine_name}': {e}", exc_info=True)
except Exception as e:
    log.critical(f"Critical error during engine initialization: {e}", exc_info=True)
    raise

log.info(f"Engine initialization completed. Loaded engines: {list(ENGINES.keys())}")

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

        if not engine or engine not in ENGINES:
            default_engine = "openai"
            if default_engine in ENGINES:
                app.state.config.ENGINE = default_engine
                log.warning(f"Engine was missing or invalid. Defaulting to '{default_engine}'.")
            else:
                log.error(f"Default engine '{default_engine}' is not available in the enabled engines.")
    except Exception as e:
        log.exception(f"Unexpected error during default engine enforcement: {e}")
        app.state.config.ENGINE = "openai"  # Fallback to 'openai'

# Call the function during app initialization, after engines are loaded
enforce_default_engine()

# Custom Exception Handlers

@app.exception_handler(RequestValidationError)
def validation_exception_handler(request: Request, exc: RequestValidationError):
    log.warning(f"Validation error for request {request.method} {request.url}: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "message": "Validation error",
            "errors": exc.errors(),
            "body": exc.body,
        },
    )

# Utility function to fetch and sanitize engine configurations
def get_populated_configs(engines: Dict[str, BaseImageEngine]) -> Dict[str, Dict[str, Any]]:
    """
    Dynamically fetch all configurations from engines,
    substituting missing or None values with empty strings.

    Args:
        engines (Dict[str, BaseImageEngine]): A dictionary of engine instances.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary containing sanitized configurations for each engine.
    """
    populated_config = {}
    for engine_key, engine_instance in engines.items():
        engine_config = engine_instance.get_config()
        # Substitute None with ""
        clean_config = {k: (v if v is not None else "") for k, v in engine_config.items()}
        populated_config[engine_key] = clean_config
    return populated_config

# Routes

@app.get("/config")
def get_config(user=Depends(get_admin_user)):
    """
    Retrieve current configuration, dynamically flattening engine-specific details.
    """
    log.debug("Retrieving current configuration.")
    general_config = {
        "enabled": getattr(app.state.config, "ENABLED", False),
        "engine": getattr(app.state.config, "ENGINE", ""),
    }

    # Dynamically fetch and sanitize engine configurations
    engine_configs = get_populated_configs(ENGINES)

    # Combine general and engine-specific configurations into a single top-level structure
    combined_config = {**general_config, **engine_configs}

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

            app.state.config.ENGINE = new_engine
            log.debug(f"Set ENGINE to '{new_engine}'.")

        # Update active engine configurations
        active_engine = ENGINES.get(app.state.config.ENGINE)
        if active_engine:
            engine_config = data.get(app.state.config.ENGINE, {})
            if isinstance(engine_config, dict):
                active_engine.update_config_in_app(engine_config, app.state.config)
                log.debug(f"Updated config for engine '{app.state.config.ENGINE}'.")

        # Populate engine-specific configs
        engine_configs = get_populated_configs(ENGINES)

        # Validate the active engine's configuration
        warnings = []
        if active_engine:
            is_valid, missing_fields = active_engine.validate_config()
            if not is_valid:
                warnings.append(f"Engine '{app.state.config.ENGINE}' is not fully configured. Missing: {', '.join(missing_fields)}.")

        # Return a complete response with warnings
        combined_config = {
            "enabled": app.state.config.ENABLED,
            "engine": app.state.config.ENGINE,
            **engine_configs,
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
            log.debug(f"Setting MODEL to '{data['model']}'")
            set_image_model(data["model"])

        # Update IMAGE_SIZE
        if "image_size" in data and data["image_size"]:
            size_pattern = r"^\d+x\d+$"
            if re.match(size_pattern, data["image_size"]):
                app.state.config.IMAGE_SIZE = data["image_size"]
                log.debug(f"Set IMAGE_SIZE to '{data['image_size']}'")
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

        # After updating configurations, validate the active engine's configuration
        engine = getattr(app.state.config, "ENGINE", "").lower()
        engine_instance: Optional[BaseImageEngine] = ENGINES.get(engine)
        if not engine_instance:
            log.error(f"Engine '{engine}' is not supported.")
            raise HTTPException(
                status_code=400,
                detail=f"Engine '{engine}' is not supported."
            )
        is_valid, missing_fields = engine_instance.validate_config()
        if not is_valid:
            log.error(f"Validation failed for engine '{engine}'. Missing: {', '.join(missing_fields)}.")
            raise HTTPException(
                status_code=400,
                detail=f"Validation failed for engine '{engine}'. Missing: {', '.join(missing_fields)}."
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
    engine = getattr(app.state.config, "ENGINE", "").lower()
    engine_instance = ENGINES.get(engine)

    if not engine_instance:
        log.warning(f"Engine '{engine}' not found.")
        return {"message": f"Engine '{engine}' is not supported."}

    if not engine_instance.is_configured():
        log.warning(f"Engine '{engine}' is not properly configured.")
        return {"message": f"Engine '{engine}' is not configured."}

    try:
        engine_instance.verify_url()
        log.info(f"Engine '{engine}' verified successfully.")
        return {"message": f"Engine '{engine}' verified successfully."}
    except Exception as e:
        log.warning(f"Failed to verify engine '{engine}': {e}")
        return {"message": f"Engine '{engine}' is unavailable. Connectivity check failed."}

@app.get("/models")
def get_models(user=Depends(get_verified_user)):
    """
    Retrieve models available in the selected engine.
    """
    engine = getattr(app.state.config, "ENGINE", "").lower()
    engine_instance: Optional[BaseImageEngine] = ENGINES.get(engine)

    log.debug(f"Fetching available models for engine '{engine}'")

    if not engine_instance:
        log.error(f"Engine '{engine}' is not supported.")
        return {
            "status": "error",
            "message": f"Engine '{engine}' is not supported. Please choose a valid engine.",
            "models": []
        }

    # Validate configuration
    is_valid, missing_fields = engine_instance.validate_config()
    if not is_valid:
        log.warning(f"Engine '{engine}' is not fully configured. Missing: {', '.join(missing_fields)}.")
        return {
            "status": "warning",
            "message": f"Engine '{engine}' is not fully configured. Missing: {', '.join(missing_fields)}",
            "models": []
        }

    try:
        models = engine_instance.list_models()
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
    engine_instance: Optional[BaseImageEngine] = ENGINES.get(engine)

    log.debug(f"Generating images using engine '{engine}' with data: {form_data.dict()}")

    if not engine_instance:
        log.error(f"Engine '{engine}' is not supported.")
        raise HTTPException(status_code=400, detail=f"Engine '{engine}' is not supported.")

    # Check if engine is configured
    if not engine_instance.is_configured():
        log.error(f"Engine '{engine}' is not properly configured.")
        raise HTTPException(status_code=400, detail=f"Engine '{engine}' is not properly configured.")

    size = form_data.size or getattr(app.state.config, "IMAGE_SIZE", "")
    log.debug(f"Using image size: '{size}'")

    try:
        images = engine_instance.generate_image(
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

        image_config = {
            "MODEL": model,
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
    engine_instance: Optional[BaseImageEngine] = ENGINES.get(engine)

    log.debug(f"Setting image model to '{model}' for engine '{engine}'")

    if not engine_instance:
        log.error(f"Engine '{engine}' is not supported.")
        raise HTTPException(status_code=400, detail=f"Engine '{engine}' is not supported.")

    # Assuming engines have a `set_model` method
    if hasattr(engine_instance, 'set_model'):
        try:
            engine_instance.set_model(model)
            log.debug(f"Model set to '{model}' successfully for engine '{engine}'")
        except Exception as e:
            log.error(f"Failed to set model for engine '{engine}': {e}")
            raise HTTPException(status_code=500, detail=f"Failed to set model for engine '{engine}': {e}")
    else:
        log.warning(f"Engine '{engine}' does not implement set_model method.")
        raise HTTPException(status_code=400, detail=f"Engine '{engine}' does not support model updates.")

    # Update the configuration directly
    app.state.config.MODEL = model
    log.debug(f"Set MODEL to '{model}'")

def get_image_model():
    """
    Get the current image model for the selected engine.
    """
    engine = getattr(app.state.config, "ENGINE", "").lower()
    engine_instance: Optional[BaseImageEngine] = ENGINES.get(engine)

    log.debug(f"Retrieving current image model for engine '{engine}'")

    if not engine_instance:
        log.error(f"Engine '{engine}' is not supported.")
        raise HTTPException(status_code=400, detail=f"Engine '{engine}' is not supported.")

    # Assuming engines have a `get_model` method
    if hasattr(engine_instance, 'get_model'):
        try:
            model = engine_instance.get_model()
            log.debug(f"Current model for engine '{engine}': '{model}'")
            return model
        except Exception as e:
            log.error(f"Failed to get model for engine '{engine}': {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get model for engine '{engine}': {e}")
    else:
        log.warning(f"Engine '{engine}' does not implement get_model method.")
        return getattr(app.state.config, "MODEL", "")
