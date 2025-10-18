"""
MLflow-based model prediction service for the Real Estate Price Prediction API V2.

This module extends the base HousePricePredictor to load champion models from MLflow
instead of pickle files.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mlflow
import mlflow.sklearn

from .config import settings
from .exceptions import ModelLoadError
from .predictor import HousePricePredictor


class MLflowModelLoadError(ModelLoadError):
    """Exception raised when MLflow model loading fails."""

    pass


class MLflowHousePricePredictor(HousePricePredictor):
    """
    MLflow-based house price prediction service.

    Extends the base HousePricePredictor to load champion models from MLflow
    experiments instead of pickle files.
    """

    model: Any  # Override parent type to allow MLflow models

    def __init__(self):
        """Initialize the predictor with MLflow champion model and demographic data."""
        self.champion_run_info: Optional[Dict] = None
        super().__init__()

    def _load_model(self) -> None:
        """Load the model from MLflow using the configured URI."""
        try:
            # Check if champion model URI is configured
            if not settings.champion_model_mlflow_uri:
                raise MLflowModelLoadError(
                    "CHAMPION_MODEL_MLFLOW_URI not configured. V2 API requires MLflow model URI."
                )

            # Setup MLflow tracking
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

            model_uri = settings.champion_model_mlflow_uri
            print(f"Loading model from: {model_uri}")

            # Load the model from the configured URI
            self.model = mlflow.sklearn.load_model(model_uri)

            print(f"✓ Successfully loaded model from {model_uri}")

            # Parse model information from URI
            if model_uri.startswith("models:/"):
                # Format: models:/model_name/version
                uri_parts = model_uri.replace("models:/", "").split("/")
                model_name = uri_parts[0] if len(uri_parts) > 0 else "unknown"
                model_version = uri_parts[1] if len(uri_parts) > 1 else "unknown"
                source = "model_registry"
            else:
                # Other URI formats (runs:/, etc.)
                model_name = "unknown"
                model_version = "unknown"
                source = "mlflow_uri"

            # Store model information for compatibility
            self.champion_run_info = {
                "model_name": model_name,
                "model_version": model_version,
                "model_uri": model_uri,
                "model_type": "mlflow_model",
                "source": source,
            }

            print(f"Model loaded: {model_name} (version: {model_version})")

            # Load model features - fallback to local features file
            try:
                features_path = Path(settings.model_features_path)
                with open(features_path, "r") as f:
                    self.model_features = json.load(f)
                print(f"    Loaded {len(self.model_features)} features from local file")
            except Exception as e:
                print(f"    Warning: Could not load features from local file: {e}")
                # Set some default features if file is not available
                self.model_features = []

            if not self.model:
                raise MLflowModelLoadError("Model loading failed")

        except Exception as e:
            if isinstance(e, MLflowModelLoadError):
                raise
            raise MLflowModelLoadError(
                f"Unexpected error loading model from registry: {e}"
            )

    def get_model_info(self) -> Dict[str, Union[str, int, List[str]]]:
        """
        Get information about the loaded MLflow champion model.

        Returns:
            Dictionary containing model metadata with correct types for Pydantic
        """
        return {
            "model_type": self.champion_run_info.get("model_type", "Unknown")
            if self.champion_run_info
            else "Unknown",
            "model_version": "2.0.0",
            "total_features": len(self.model_features) if self.model_features else 0,
            "features": self.model_features or [],
            "demographics_zipcodes": len(self.demographics_data)
            if self.demographics_data is not None
            else 0,
        }


# Global MLflow predictor instance - only instantiate when needed
mlflow_predictor = None


def get_mlflow_predictor():
    """Get or create the MLflow predictor instance."""
    global mlflow_predictor
    if mlflow_predictor is None:
        mlflow_predictor = MLflowHousePricePredictor()
    return mlflow_predictor
