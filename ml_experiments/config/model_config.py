"""
Model configuration settings.

This module centralizes all model-related configuration parameters
that were scattered across multiple experiment scripts.
"""

from typing import Any, Dict, List, Optional

# Best parameters from hyperparameter tuning (from original experiments)
BEST_LIGHTGBM_PARAMS = {
    "n_estimators": 50,
    "max_depth": 9,
    "learning_rate": 0.1,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 1.5,
    "num_leaves": 31,
    "random_state": 42,
    "n_jobs": -1,
    "objective": "regression",
    "metric": "rmse",
    "verbose": -1,
}

BEST_XGBOOST_PARAMS = {
    "n_estimators": 50,
    "max_depth": 6,
    "learning_rate": 0.2,
    "subsample": 1.0,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
}

BEST_KNN_PARAMS = {
    "n_neighbors": 5,
    "weights": "uniform",
    "algorithm": "auto",
    "leaf_size": 30,
    "p": 2,
    "metric": "minkowski",
}

# Default model parameters
DEFAULT_MODEL_PARAMS = {
    "knn": BEST_KNN_PARAMS,
    "xgboost": BEST_XGBOOST_PARAMS,
    "lightgbm": BEST_LIGHTGBM_PARAMS,
}

# Hyperparameter tuning grids
HYPERPARAMETER_GRIDS = {
    "knn": {
        "n_neighbors": [3, 5, 7, 10, 15],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree"],
        "leaf_size": [20, 30, 40],
        "p": [1, 2],
    },
    "xgboost": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 6, 9],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [1, 1.5, 2.0],
    },
    "lightgbm": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 6, 9],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [1, 1.5, 2.0],
        "num_leaves": [31, 50, 100],
    },
}

# Conservative hyperparameter grids (smaller search space)
CONSERVATIVE_HYPERPARAMETER_GRIDS = {
    "knn": {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto"],
        "leaf_size": [30],
        "p": [2],
    },
    "xgboost": {
        "n_estimators": [50, 100],
        "max_depth": [3, 6],
        "learning_rate": [0.1, 0.2],
        "subsample": [0.9, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_alpha": [0, 0.1],
        "reg_lambda": [1, 1.5],
    },
    "lightgbm": {
        "n_estimators": [50, 100],
        "max_depth": [3, 6],
        "learning_rate": [0.1, 0.2],
        "subsample": [0.9, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_alpha": [0, 0.1],
        "reg_lambda": [1, 1.5],
        "num_leaves": [31, 50],
    },
}

# Model-specific fixed parameters (always applied)
MODEL_FIXED_PARAMS = {
    "knn": {},  # No fixed params for KNN
    "xgboost": {
        "random_state": 42,
        "n_jobs": -1,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
    },
    "lightgbm": {
        "random_state": 42,
        "n_jobs": -1,
        "objective": "regression",
        "metric": "rmse",
        "verbose": -1,
    },
}

# Supported model types
SUPPORTED_MODELS = ["knn", "xgboost", "lightgbm"]

# Model aliases for convenience
MODEL_ALIASES = {
    "k-nn": "knn",
    "kneighbors": "knn",
    "xgb": "xgboost",
    "lgb": "lightgbm",
    "lgbm": "lightgbm",
}

# Model performance expectations (for validation)
MODEL_PERFORMANCE_EXPECTATIONS = {
    "knn": {
        "min_r2": 0.3,  # KNN typically performs worse
        "max_training_time": 60,  # seconds
    },
    "xgboost": {
        "min_r2": 0.6,  # XGBoost should perform well
        "max_training_time": 300,  # seconds
    },
    "lightgbm": {
        "min_r2": 0.6,  # LightGBM should perform well
        "max_training_time": 180,  # seconds
    },
}


def get_model_params(model_type: str, param_set: str = "best") -> Dict[str, Any]:
    """Get model parameters for a specific model type.

    Args:
        model_type: Type of model ("knn", "xgboost", "lightgbm")
        param_set: Parameter set to use ("best", "default")

    Returns:
        Dictionary with model parameters

    Raises:
        ValueError: If model_type is not supported
    """
    # Resolve aliases
    model_type = MODEL_ALIASES.get(model_type, model_type)

    if model_type not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model type: {model_type}. Supported: {SUPPORTED_MODELS}"
        )

    if param_set == "best" or param_set == "default":
        params = DEFAULT_MODEL_PARAMS[model_type].copy()
    else:
        raise ValueError(
            f"Unknown parameter set: {param_set}. Available: ['best', 'default']"
        )

    return params


def get_hyperparameter_grid(
    model_type: str, grid_type: str = "full"
) -> Dict[str, List]:
    """Get hyperparameter grid for tuning.

    Args:
        model_type: Type of model
        grid_type: Type of grid ("full", "conservative")

    Returns:
        Dictionary with hyperparameter grid

    Raises:
        ValueError: If model_type is not supported
    """
    # Resolve aliases
    model_type = MODEL_ALIASES.get(model_type, model_type)

    if model_type not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model type: {model_type}. Supported: {SUPPORTED_MODELS}"
        )

    if grid_type == "full":
        return HYPERPARAMETER_GRIDS[model_type].copy()
    elif grid_type == "conservative":
        return CONSERVATIVE_HYPERPARAMETER_GRIDS[model_type].copy()
    else:
        raise ValueError(
            f"Unknown grid type: {grid_type}. Available: ['full', 'conservative']"
        )


def get_fixed_params(model_type: str) -> Dict[str, Any]:
    """Get fixed parameters that should always be applied.

    Args:
        model_type: Type of model

    Returns:
        Dictionary with fixed parameters
    """
    # Resolve aliases
    model_type = MODEL_ALIASES.get(model_type, model_type)

    if model_type not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model type: {model_type}. Supported: {SUPPORTED_MODELS}"
        )

    return MODEL_FIXED_PARAMS[model_type].copy()


def merge_model_params(
    model_type: str, base_params: Dict[str, Any], override_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge model parameters with fixed parameters and overrides.

    Args:
        model_type: Type of model
        base_params: Base parameters
        override_params: Parameters to override

    Returns:
        Merged parameters dictionary
    """
    # Start with base parameters
    merged_params = base_params.copy()

    # Apply fixed parameters (these cannot be overridden)
    fixed_params = get_fixed_params(model_type)
    merged_params.update(fixed_params)

    # Apply override parameters
    merged_params.update(override_params)

    return merged_params


def validate_model_params(model_type: str, params: Dict[str, Any]) -> None:
    """Validate model parameters.

    Args:
        model_type: Type of model
        params: Parameters to validate

    Raises:
        ValueError: If parameters are invalid
    """
    # Resolve aliases
    model_type = MODEL_ALIASES.get(model_type, model_type)

    if model_type not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model type: {model_type}. Supported: {SUPPORTED_MODELS}"
        )

    # Model-specific validation
    if model_type == "knn":
        if "n_neighbors" in params and params["n_neighbors"] <= 0:
            raise ValueError("n_neighbors must be positive")

    elif model_type in ["xgboost", "lightgbm"]:
        if "n_estimators" in params and params["n_estimators"] <= 0:
            raise ValueError("n_estimators must be positive")

        if "learning_rate" in params and not (0 < params["learning_rate"] <= 1):
            raise ValueError("learning_rate must be between 0 and 1")

        if "max_depth" in params and params["max_depth"] <= 0:
            raise ValueError("max_depth must be positive")


def get_model_performance_expectation(model_type: str) -> Dict[str, Any]:
    """Get performance expectations for a model type.

    Args:
        model_type: Type of model

    Returns:
        Dictionary with performance expectations
    """
    # Resolve aliases
    model_type = MODEL_ALIASES.get(model_type, model_type)

    if model_type not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model type: {model_type}. Supported: {SUPPORTED_MODELS}"
        )

    return MODEL_PERFORMANCE_EXPECTATIONS[model_type].copy()


class ModelConfig:
    """Model configuration class for experiments."""

    def __init__(
        self,
        model_type: str,
        param_set: str = "best",
        override_params: Optional[Dict[str, Any]] = None,
        use_feature_selection: bool = False,
        feature_selection_k: int = 50,
    ):
        """Initialize model configuration.

        Args:
            model_type: Type of model
            param_set: Parameter set to use
            override_params: Parameters to override
            use_feature_selection: Whether to use feature selection
            feature_selection_k: Number of features to select
        """
        # Resolve aliases
        self.model_type = MODEL_ALIASES.get(model_type, model_type)

        if self.model_type not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model type: {model_type}. Supported: {SUPPORTED_MODELS}"
            )

        # Get base parameters
        base_params = get_model_params(self.model_type, param_set)

        # Merge with overrides
        override_params = override_params or {}
        self.params = merge_model_params(self.model_type, base_params, override_params)

        # Feature selection settings
        self.use_feature_selection = use_feature_selection
        self.feature_selection_k = feature_selection_k

        # Validate parameters
        validate_model_params(self.model_type, self.params)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        return {
            "model_type": self.model_type,
            "params": self.params,
            "use_feature_selection": self.use_feature_selection,
            "feature_selection_k": self.feature_selection_k,
        }

    def get_tuning_grid(self, grid_type: str = "full") -> Dict[str, List]:
        """Get hyperparameter tuning grid for this model.

        Args:
            grid_type: Type of grid to return

        Returns:
            Hyperparameter grid
        """
        return get_hyperparameter_grid(self.model_type, grid_type)

    def get_performance_expectation(self) -> Dict[str, Any]:
        """Get performance expectations for this model.

        Returns:
            Performance expectations
        """
        return get_model_performance_expectation(self.model_type)

    def create_variant(self, override_params: Dict[str, Any]) -> "ModelConfig":
        """Create a variant of this configuration with different parameters.

        Args:
            override_params: Parameters to override

        Returns:
            New ModelConfig instance
        """
        return ModelConfig(
            model_type=self.model_type,
            param_set="best",  # Use current params as base
            override_params={**self.params, **override_params},
            use_feature_selection=self.use_feature_selection,
            feature_selection_k=self.feature_selection_k,
        )
