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
        """Load the model from MLflow Model Registry."""
        try:
            # Setup MLflow tracking
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

            # Use Model Registry approach (hardcoded as requested)
            model_name = "docker_lightgbm_house_price_model"
            model_version = "latest"
            model_uri = f"models:/{model_name}/{model_version}"

            print(f"Loading model from Model Registry: {model_uri}")

            # Load the model from the Model Registry
            self.model = mlflow.sklearn.load_model(model_uri)

            print(f"✓ Successfully loaded model from {model_uri}")

            # Store model information for compatibility
            self.champion_run_info = {
                "model_name": model_name,
                "model_version": model_version,
                "model_uri": model_uri,
                "model_type": "lightgbm",  # Known from the model name
                "source": "model_registry",
            }

            print(
                f"Model loaded from Model Registry: {model_name} (version: {model_version})"
            )

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

        # COMMENTED OUT: Original experiment search approach
        # This was causing issues with artifact downloads
        """
        # Setup MLflow tracking
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = MlflowClient()

        # Search runs in the specified experiment
        runs = client.search_runs(
            experiment_ids=[settings.mlflow_experiment_id],
            filter_string="status = 'FINISHED'",
            order_by=[f"metrics.{settings.champion_metric} DESC"],
            max_results=50,  # Get more runs to find the best one
        )

        if not runs:
            raise MLflowModelLoadError(
                f"No finished runs found in experiment {settings.mlflow_experiment_id}"
            )

        print(
            f"Found {len(runs)} finished runs in experiment {settings.mlflow_experiment_id}"
        )

        # Try to load models from runs in order of best metric score
        champion_run = None
        model_loaded = False

        for i, run in enumerate(runs):
            run_id = run.info.run_id
            test_r2 = run.data.metrics.get(settings.champion_metric, 0.0)
            model_type = run.data.tags.get("model_type", "Unknown")

            print(
                f"  Trying run {i + 1}: {run_id} ({model_type}, {settings.champion_metric}: {test_r2:.4f})"
            )

            try:
                # Use the proven approach: load directly with runs:/ URI
                model_uri = f"runs:/{run_id}/model"
                self.model = mlflow.sklearn.load_model(model_uri)

                print(f"    ✓ Successfully loaded model from {model_uri}")
                champion_run = run
                model_loaded = True
                break

            except Exception as e:
                print(f"    ✗ Failed to load model: {e}")
                continue

        if not model_loaded or not champion_run:
            raise MLflowModelLoadError(
                f"Could not load any model from experiment {settings.mlflow_experiment_id}. "
                f"Tried {len(runs)} runs."
            )

        run_id = champion_run.info.run_id

        # Store champion run information
        self.champion_run_info = {
            "run_id": run_id,
            "experiment_id": settings.mlflow_experiment_id,
            "test_r2": champion_run.data.metrics.get(settings.champion_metric, 0.0),
            "test_mae": champion_run.data.metrics.get("test_mae", 0.0),
            "test_rmse": champion_run.data.metrics.get("test_rmse", 0.0),
            "train_r2": champion_run.data.metrics.get("train_r2", 0.0),
            "model_type": champion_run.data.tags.get("model_type", "Unknown"),
            "model_uri": f"runs:/{run_id}/model",
        }

        print(
            f"Champion model loaded: {self.champion_run_info['model_type']} "
            f"(test_r2: {self.champion_run_info['test_r2']:.4f})"
        )

        # Load model features - try MLflow artifacts first, then fallback
        try:
            # Try to get features from MLflow run data/tags
            features_tag = champion_run.data.tags.get("features")
            if features_tag:
                self.model_features = json.loads(features_tag)
                print(
                    f"    Loaded {len(self.model_features)} features from run tags"
                )
            else:
                raise Exception("No features in run tags")

        except Exception:
            try:
                # Try to download feature_info.json artifact
                artifact_path = client.download_artifacts(
                    run_id=run_id, path="feature_info.json"
                )
                with open(artifact_path, "r") as f:
                    feature_data = json.load(f)
                    self.model_features = feature_data.get("features", [])
                print(
                    f"    Loaded {len(self.model_features)} features from artifacts"
                )

            except Exception:
                # Final fallback to local features file
                features_path = Path(settings.model_features_path)
                with open(features_path, "r") as f:
                    self.model_features = json.load(f)
                print(
                    f"    Loaded {len(self.model_features)} features from local file"
                )

        if not self.model or not self.model_features:
            raise MLflowModelLoadError("Champion model or features list is empty")
        """

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


# Global MLflow predictor instance
mlflow_predictor = MLflowHousePricePredictor()
