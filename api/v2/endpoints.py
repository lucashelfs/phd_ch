"""
API V2 endpoints for MLflow-based model serving.

This module provides V2 endpoints that use MLflow champion models
instead of pickle files, while maintaining the same API structure as V1.
"""

from fastapi import APIRouter, HTTPException, status

from ..core.exceptions import DemographicDataError, PredictionError
from ..core.mlflow_predictor import MLflowModelLoadError, get_mlflow_predictor
from ..v1.models import (
    HouseFeaturesRequest,
    MinimalHouseFeaturesRequest,
    ModelInfoResponse,
    PredictionResponse,
)

# Create V2 router
router = APIRouter(prefix="/v2", tags=["v2"])


@router.get("/info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Get information about the loaded MLflow champion model.

    Returns detailed information about the champion model including
    MLflow run details and performance metrics.
    """
    try:
        predictor = get_mlflow_predictor()
        model_info = predictor.get_model_info()
        return ModelInfoResponse(**model_info)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "MODEL_INFO_ERROR",
                "message": f"Failed to get model information: {str(e)}",
                "details": {},
            },
        )


@router.post("/predict", response_model=PredictionResponse)
async def predict_house_price(request: HouseFeaturesRequest):
    """
    Predict house price using the MLflow champion model.

    This endpoint uses the champion model (highest test R² score) from the
    specified MLflow experiment to make predictions with demographic enrichment.

    Args:
        request: House features including zipcode for demographic enrichment

    Returns:
        Prediction result with MLflow metadata

    Raises:
        HTTPException: If prediction fails due to missing data or model errors
    """
    try:
        # Convert request to dictionary
        house_data = request.dict()

        # Make prediction using MLflow predictor
        predictor = get_mlflow_predictor()
        result = predictor.predict(house_data)

        return PredictionResponse(**result)

    except DemographicDataError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "DEMOGRAPHIC_DATA_ERROR",
                "message": str(e),
                "details": getattr(e, "details", {}),
            },
        )
    except PredictionError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "PREDICTION_ERROR",
                "message": str(e),
                "details": getattr(e, "details", {}),
            },
        )
    except MLflowModelLoadError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "MLFLOW_MODEL_ERROR", "message": str(e), "details": {}},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_ERROR",
                "message": f"Unexpected error during prediction: {str(e)}",
                "details": {},
            },
        )


@router.post("/predict/minimal", response_model=PredictionResponse)
async def predict_house_price_minimal(request: MinimalHouseFeaturesRequest):
    """
    Predict house price using only core features with the MLflow champion model.

    This endpoint uses only the core house features (bedrooms, bathrooms, etc.)
    for prediction, excluding most demographic features. Still requires zipcode
    for demographic enrichment.

    Args:
        request: Core house features and zipcode

    Returns:
        Prediction result with minimal features metadata

    Raises:
        HTTPException: If prediction fails due to missing data or model errors
    """
    try:
        # Convert request to dictionary
        house_data = request.dict()

        # Make minimal prediction using MLflow predictor
        predictor = get_mlflow_predictor()
        result = predictor.predict_minimal(house_data)

        return PredictionResponse(**result)

    except DemographicDataError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "DEMOGRAPHIC_DATA_ERROR",
                "message": str(e),
                "details": getattr(e, "details", {}),
            },
        )
    except PredictionError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "PREDICTION_ERROR",
                "message": str(e),
                "details": getattr(e, "details", {}),
            },
        )
    except MLflowModelLoadError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "MLFLOW_MODEL_ERROR", "message": str(e), "details": {}},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_ERROR",
                "message": f"Unexpected error during minimal prediction: {str(e)}",
                "details": {},
            },
        )
