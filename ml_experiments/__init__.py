"""
ML Experiments Package

A modular and production-ready machine learning experimentation framework
for real estate price prediction. This package consolidates and organizes
the previously scattered experiment code into reusable components.

Key Features:
- Centralized configuration management
- Reusable model factories and pipelines
- Comprehensive evaluation metrics
- MLflow integration for experiment tracking
- Data processing utilities

Usage:
    from ml_experiments.core import models, evaluation, data_processing
    from ml_experiments.config import data_config, model_config, mlflow_config

    # Create a model configuration
    model_cfg = model_config.ModelConfig("xgboost", param_set="best")

    # Create a data configuration
    data_cfg = data_config.DataConfig(scaler_type="robust", test_size=0.2)

    # Create a model pipeline
    pipeline = models.create_model_pipeline(
        model_type=model_cfg.model_type,
        model_params=model_cfg.params,
        scaler_type=data_cfg.scaler_type
    )
"""

__version__ = "1.0.0"
__author__ = "ML Engineering Team"

# Import main modules for convenience
from . import config, core
from .config.cv_config import CrossValidationConfig
from .config.data_config import DataConfig
from .config.mlflow_config import MLflowConfig
from .config.model_config import ModelConfig
from .core.data_processing import DataProcessor, load_and_prepare_data
from .core.evaluation import (
    ModelEvaluator,
    calculate_overfitting_metrics,
)
from .core.mlflow_utils import setup_docker_mlflow_environment, setup_mlflow

# Make key classes and functions easily accessible
from .core.models import ModelFactory, create_model_pipeline, get_quick_models

__all__ = [
    # Modules
    "core",
    "config",
    # Core classes and functions
    "ModelFactory",
    "create_model_pipeline",
    "get_quick_models",
    "ModelEvaluator",
    "calculate_overfitting_metrics",
    "DataProcessor",
    "load_and_prepare_data",
    "setup_docker_mlflow_environment",
    "setup_mlflow",
    # Configuration classes
    "DataConfig",
    "ModelConfig",
    "MLflowConfig",
    "CrossValidationConfig",
]
