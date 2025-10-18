"""
Main FastAPI application for the Real Estate Price Prediction API.

This module creates and configures the FastAPI application with
all endpoints, middleware, and error handling.
"""

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .core.config import settings
from .core.exceptions import APIException
from .core.predictor import predictor
from .v1.endpoints import router as v1_router
from .core.models import ErrorResponse, HealthResponse

# Create FastAPI application
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    debug=settings.debug,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Global exception handlers
@app.exception_handler(APIException)
async def api_exception_handler(request: Request, exc: APIException):
    """Handle custom API exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.error_code, message=exc.message, details=exc.details
        ).dict(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="VALIDATION_ERROR",
            message="Request validation failed",
            details={"validation_errors": exc.errors()},
        ).dict(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    # If detail is already a dict (from our endpoints), use it directly
    if isinstance(exc.detail, dict):
        error_content = ErrorResponse(
            error=exc.detail.get("error", "HTTP_ERROR"),
            message=exc.detail.get("message", str(exc.detail)),
            details=exc.detail.get("details", {}),
        ).dict()
    else:
        error_content = ErrorResponse(
            error="HTTP_ERROR",
            message=str(exc.detail),
            details={"status_code": exc.status_code},
        ).dict()

    return JSONResponse(status_code=exc.status_code, content=error_content)


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    error_response = ErrorResponse(
        error="INTERNAL_ERROR",
        message="An unexpected error occurred",
        details={"exception_type": type(exc).__name__} if settings.debug else {},
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=error_response.dict()
    )


# Determine available versions
def get_available_versions():
    """Get list of available API versions."""
    versions = ["v1"]
    if settings.champion_model_mlflow_uri:
        versions.append("v2")
    return versions


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """
    API root endpoint with basic information.

    Returns API metadata and available versions.
    """
    available_versions = get_available_versions()
    endpoints = {
        "health": "/health",
        "versions": "/versions",
        "docs": "/docs" if settings.debug else "disabled",
        "v1": {
            "info": "/v1/info",
            "predict": "/v1/predict",
            "predict_minimal": "/v1/predict/minimal",
        },
    }

    if "v2" in available_versions:
        endpoints["v2"] = {
            "info": "/v2/info",
            "predict": "/v2/predict",
            "predict_minimal": "/v2/predict/minimal",
        }

    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "description": settings.api_description,
        "available_versions": available_versions,
        "endpoints": endpoints,
    }


# Versions endpoint
@app.get("/versions", tags=["root"])
async def get_versions():
    """
    Get information about available API versions and their status.

    Returns detailed information about which versions are active
    and what type of models they use.
    """
    versions_info = {
        "available_versions": get_available_versions(),
        "v1": {
            "status": "active",
            "model_type": "pickle",
            "description": "Original API using pickle-based models",
        },
    }

    if settings.champion_model_mlflow_uri:
        versions_info["v2"] = {
            "status": "active",
            "model_type": "mlflow",
            "model_uri": settings.champion_model_mlflow_uri,
            "description": "MLflow-based API using champion models",
        }
    else:
        versions_info["v2"] = {
            "status": "disabled",
            "reason": "CHAMPION_MODEL_MLFLOW_URI not configured",
        }

    return versions_info


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.

    Returns service status and component health information.
    """
    try:
        # Check if model and demographics are loaded
        model_loaded = predictor.model is not None
        demographics_loaded = predictor.demographics_data is not None

        # Determine overall status
        if model_loaded and demographics_loaded:
            status_value = "healthy"
        elif model_loaded:
            status_value = "degraded"  # Model loaded but no demographics
        else:
            status_value = "unhealthy"  # Model not loaded

        return HealthResponse(
            status=status_value,
            version=settings.api_version,
            model_loaded=model_loaded,
            demographics_loaded=demographics_loaded,
        )

    except Exception:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=HealthResponse(
                status="unhealthy",
                version=settings.api_version,
                model_loaded=False,
                demographics_loaded=False,
            ).dict(),
        )


# Include v1 router (always available)
app.include_router(v1_router)

# Conditionally include v2 router only if MLflow URI is configured
if settings.champion_model_mlflow_uri:
    from .v2.endpoints import router as v2_router

    app.include_router(v2_router)
    print("V2 API enabled with MLflow model")
else:
    print("V2 API disabled - CHAMPION_MODEL_MLFLOW_URI not configured")


# Startup event
@app.on_event("startup")
async def startup_event():
    """
    Application startup event.

    Performs initialization tasks and validates system readiness.
    """
    print(f"Starting {settings.api_title} v{settings.api_version}")

    # Validate model and data loading
    try:
        model_info = predictor.get_model_info()
        print(
            f"Model loaded: {model_info['model_type']} with {model_info['total_features']} features"
        )
        print(
            f"Demographics data: {model_info['demographics_zipcodes']} zipcodes available"
        )
    except Exception as e:
        print(f"Warning: Model validation failed: {e}")

    print("API startup complete")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """
    Application shutdown event.

    Performs cleanup tasks before application termination.
    """
    print(f"Shutting down {settings.api_title}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=settings.workers if not settings.debug else 1,
    )
