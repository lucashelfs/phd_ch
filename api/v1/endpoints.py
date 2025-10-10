"""
API v1 endpoints for the Real Estate Price Prediction API.

This module defines the FastAPI endpoints for version 1 of the API,
including prediction endpoints and model information.
"""

from typing import Dict, Any

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from ..core.config import settings
from ..core.predictor import predictor
from ..core.exceptions import APIException, PredictionError, DemographicDataError
from .models import (
    HouseFeaturesRequest,
    MinimalHouseFeaturesRequest,
    PredictionResponse,
    ModelInfoResponse,
    ErrorResponse
)

# Create router for v1 endpoints
router = APIRouter(prefix="/v1", tags=["v1"])


@router.get("/info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Get information about the loaded model.
    
    Returns model metadata including type, version, features, and
    demographic data availability.
    """
    try:
        model_info = predictor.get_model_info()
        return ModelInfoResponse(**model_info)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model information: {str(e)}"
        )


@router.post("/predict", response_model=PredictionResponse)
async def predict_house_price(house_data: HouseFeaturesRequest):
    """
    Predict house price using all available features.
    
    This endpoint accepts comprehensive house features and returns
    a price prediction with metadata. Demographic data is automatically
    added based on the provided zipcode.
    
    Args:
        house_data: House features including location and property details
        
    Returns:
        Prediction response with price estimate and metadata
        
    Raises:
        HTTPException: For validation errors, missing demographic data,
                      or prediction failures
    """
    try:
        # Convert Pydantic model to dictionary
        house_dict = house_data.dict()
        
        # Make prediction
        prediction_result = predictor.predict(house_dict)
        
        # Return structured response
        return PredictionResponse(**prediction_result)
        
    except DemographicDataError as e:
        raise HTTPException(
            status_code=e.status_code,
            detail={
                "error": e.error_code,
                "message": e.message,
                "details": e.details
            }
        )
    except PredictionError as e:
        raise HTTPException(
            status_code=e.status_code,
            detail={
                "error": e.error_code,
                "message": e.message,
                "details": e.details
            }
        )
    except APIException as e:
        raise HTTPException(
            status_code=e.status_code,
            detail={
                "error": e.error_code,
                "message": e.message,
                "details": e.details
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_ERROR",
                "message": f"Unexpected error during prediction: {str(e)}"
            }
        )


@router.post("/predict/minimal", response_model=PredictionResponse)
async def predict_house_price_minimal(house_data: MinimalHouseFeaturesRequest):
    """
    Predict house price using only core house features (bonus endpoint).
    
    This endpoint accepts only the essential house features and returns
    a price prediction. This demonstrates the model's ability to make
    predictions with reduced feature sets while still incorporating
    demographic data based on zipcode.
    
    Args:
        house_data: Core house features (bedrooms, bathrooms, sqft, etc.)
        
    Returns:
        Prediction response with price estimate and metadata
        
    Raises:
        HTTPException: For validation errors, missing demographic data,
                      or prediction failures
    """
    try:
        # Convert Pydantic model to dictionary
        house_dict = house_data.dict()
        
        # Make minimal prediction
        prediction_result = predictor.predict_minimal(house_dict)
        
        # Return structured response
        return PredictionResponse(**prediction_result)
        
    except DemographicDataError as e:
        raise HTTPException(
            status_code=e.status_code,
            detail={
                "error": e.error_code,
                "message": e.message,
                "details": e.details
            }
        )
    except PredictionError as e:
        raise HTTPException(
            status_code=e.status_code,
            detail={
                "error": e.error_code,
                "message": e.message,
                "details": e.details
            }
        )
    except APIException as e:
        raise HTTPException(
            status_code=e.status_code,
            detail={
                "error": e.error_code,
                "message": e.message,
                "details": e.details
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_ERROR",
                "message": f"Unexpected error during minimal prediction: {str(e)}"
            }
        )
