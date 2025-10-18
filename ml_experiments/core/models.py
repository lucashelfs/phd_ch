"""
Model factories and pipeline creation utilities.

This module consolidates all model-related functionality that was duplicated
across multiple experiment scripts.
"""

from itertools import product
from typing import Any, Dict, Optional, Union

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn import neighbors, pipeline, preprocessing
from sklearn.feature_selection import SelectKBest, f_regression


def create_model_pipeline(
    model_type: str,
    model_params: Dict[str, Any],
    scaler_type: str = "robust",
    scaler_params: Optional[Dict[str, Any]] = None,
    feature_selector: Optional[Any] = None,
) -> pipeline.Pipeline:
    """Create a model pipeline with preprocessing.

    Args:
        model_type: Type of model ("knn", "xgboost", "lightgbm")
        model_params: Parameters for the model
        scaler_type: Type of scaler ("robust", "standard", "minmax")
        scaler_params: Parameters for the scaler
        feature_selector: Feature selector instance (optional)

    Returns:
        Sklearn pipeline with preprocessing and model
    """
    steps = []

    # Add scaler
    scaler = create_scaler(scaler_type, scaler_params)
    steps.append(("scaler", scaler))

    # Add feature selector if specified
    if feature_selector is not None:
        steps.append(("feature_selector", feature_selector))

    # Add model
    model = create_model(model_type, model_params)
    steps.append(("model", model))

    return pipeline.Pipeline(steps)


def create_scaler(
    scaler_type: str, scaler_params: Optional[Dict[str, Any]] = None
) -> Union[
    preprocessing.RobustScaler, preprocessing.StandardScaler, preprocessing.MinMaxScaler
]:
    """Create a scaler instance.

    Args:
        scaler_type: Type of scaler ("robust", "standard", "minmax")
        scaler_params: Parameters for the scaler

    Returns:
        Scaler instance
    """
    if scaler_params is None:
        scaler_params = {}

    if scaler_type == "robust":
        # Only use valid RobustScaler parameters
        valid_params = {"quantile_range": (25.0, 75.0)}
        # Filter scaler_params to only include valid RobustScaler parameters
        robust_scaler_params = [
            "quantile_range",
            "with_centering",
            "with_scaling",
            "copy",
            "unit_variance",
        ]
        for key, value in scaler_params.items():
            if key in robust_scaler_params:
                valid_params[key] = value
        return preprocessing.RobustScaler(**valid_params)

    elif scaler_type == "standard":
        return preprocessing.StandardScaler(**scaler_params)

    elif scaler_type == "minmax":
        return preprocessing.MinMaxScaler(**scaler_params)

    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")


def create_model(model_type: str, model_params: Dict[str, Any]) -> Any:
    """Create a model instance.

    Args:
        model_type: Type of model ("knn", "xgboost", "lightgbm")
        model_params: Parameters for the model

    Returns:
        Model instance
    """
    if model_type == "knn":
        return neighbors.KNeighborsRegressor(**model_params)

    elif model_type == "xgboost":
        return xgb.XGBRegressor(**model_params)

    elif model_type == "lightgbm":
        return lgb.LGBMRegressor(**model_params)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_feature_selector(
    selector_type: str = "kbest", k: int = 50, score_func: Any = f_regression
) -> Any:
    """Create a feature selector.

    Args:
        selector_type: Type of selector ("kbest")
        k: Number of features to select
        score_func: Scoring function for feature selection

    Returns:
        Feature selector instance
    """
    if selector_type == "kbest":
        return SelectKBest(score_func=score_func, k=k)
    else:
        raise ValueError(f"Unknown selector type: {selector_type}")


def generate_param_combinations(
    param_grid: Dict[str, list], max_combinations: int = 50, random_seed: int = 42
) -> list:
    """Generate parameter combinations from grid, limiting to max_combinations.

    Args:
        param_grid: Dictionary of parameter names and their possible values
        max_combinations: Maximum number of combinations to generate
        random_seed: Random seed for reproducible sampling

    Returns:
        List of parameter dictionaries
    """
    # Get all possible combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combinations = list(product(*values))

    # Limit combinations if too many
    if len(all_combinations) > max_combinations:
        # Use numpy random sampling for reproducibility
        np.random.seed(random_seed)
        selected_indices = np.random.choice(
            len(all_combinations), max_combinations, replace=False
        )
        selected_combinations = [all_combinations[i] for i in selected_indices]
    else:
        selected_combinations = all_combinations

    # Convert to list of dictionaries
    param_combinations = []
    for combination in selected_combinations:
        param_dict = dict(zip(keys, combination))
        param_combinations.append(param_dict)

    return param_combinations


class ModelFactory:
    """Factory class for creating models and pipelines."""

    def __init__(self):
        """Initialize model factory."""
        self.supported_models = ["knn", "xgboost", "lightgbm"]
        self.supported_scalers = ["robust", "standard", "minmax"]

    def create_pipeline(
        self,
        model_type: str,
        model_params: Dict[str, Any],
        scaler_type: str = "robust",
        scaler_params: Optional[Dict[str, Any]] = None,
        use_feature_selection: bool = False,
        feature_selection_k: int = 50,
    ) -> pipeline.Pipeline:
        """Create a complete model pipeline.

        Args:
            model_type: Type of model
            model_params: Model parameters
            scaler_type: Type of scaler
            scaler_params: Scaler parameters
            use_feature_selection: Whether to use feature selection
            feature_selection_k: Number of features to select

        Returns:
            Complete pipeline
        """
        if model_type not in self.supported_models:
            raise ValueError(
                f"Unsupported model type: {model_type}. Supported: {self.supported_models}"
            )

        if scaler_type not in self.supported_scalers:
            raise ValueError(
                f"Unsupported scaler type: {scaler_type}. Supported: {self.supported_scalers}"
            )

        # Create feature selector if needed
        feature_selector = None
        if use_feature_selection:
            feature_selector = create_feature_selector("kbest", feature_selection_k)

        return create_model_pipeline(
            model_type=model_type,
            model_params=model_params,
            scaler_type=scaler_type,
            scaler_params=scaler_params,
            feature_selector=feature_selector,
        )

    def get_default_params(self, model_type: str) -> Dict[str, Any]:
        """Get default parameters for a model type.

        Args:
            model_type: Type of model

        Returns:
            Dictionary with default parameters
        """
        if model_type == "knn":
            return {
                "n_neighbors": 5,
                "weights": "uniform",
                "algorithm": "auto",
                "leaf_size": 30,
                "p": 2,
                "metric": "minkowski",
            }

        elif model_type == "xgboost":
            return {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_alpha": 0,
                "reg_lambda": 1,
                "random_state": 42,
                "n_jobs": -1,
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
            }

        elif model_type == "lightgbm":
            return {
                "n_estimators": 100,
                "max_depth": -1,
                "learning_rate": 0.1,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_alpha": 0.0,
                "reg_lambda": 0.0,
                "num_leaves": 31,
                "random_state": 42,
                "n_jobs": -1,
                "objective": "regression",
                "metric": "rmse",
                "verbose": -1,
            }

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def get_tuning_grid(self, model_type: str) -> Dict[str, list]:
        """Get hyperparameter tuning grid for a model type.

        Args:
            model_type: Type of model

        Returns:
            Dictionary with parameter grid for tuning
        """
        if model_type == "knn":
            return {
                "n_neighbors": [3, 5, 7, 10, 15],
                "weights": ["uniform", "distance"],
                "algorithm": ["auto", "ball_tree", "kd_tree"],
                "leaf_size": [20, 30, 40],
                "p": [1, 2],
            }

        elif model_type == "xgboost":
            return {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
                "reg_alpha": [0, 0.1, 1.0],
                "reg_lambda": [1, 1.5, 2.0],
            }

        elif model_type == "lightgbm":
            return {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
                "reg_alpha": [0, 0.1, 1.0],
                "reg_lambda": [1, 1.5, 2.0],
                "num_leaves": [31, 50, 100],
            }

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def create_optimized_params(
        self, model_type: str, base_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create optimized parameters based on best practices.

        Args:
            model_type: Type of model
            base_params: Base parameters to override defaults

        Returns:
            Optimized parameter dictionary
        """
        default_params = self.get_default_params(model_type)

        if base_params:
            default_params.update(base_params)

        return default_params


# Pre-configured model instances for common use cases
def get_quick_models() -> Dict[str, pipeline.Pipeline]:
    """Get a set of pre-configured models for quick experimentation.

    Returns:
        Dictionary of model name to pipeline
    """
    factory = ModelFactory()
    models = {}

    # KNN model
    models["knn_default"] = factory.create_pipeline(
        model_type="knn",
        model_params=factory.get_default_params("knn"),
        scaler_type="robust",
    )

    # XGBoost model
    models["xgboost_default"] = factory.create_pipeline(
        model_type="xgboost",
        model_params=factory.get_default_params("xgboost"),
        scaler_type="robust",
    )

    # LightGBM model
    models["lightgbm_default"] = factory.create_pipeline(
        model_type="lightgbm",
        model_params=factory.get_default_params("lightgbm"),
        scaler_type="robust",
    )

    return models
