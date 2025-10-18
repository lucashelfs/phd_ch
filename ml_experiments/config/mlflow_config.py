"""
MLflow configuration settings.

This module centralizes all MLflow-related configuration parameters
that were scattered across multiple experiment scripts.
"""

from typing import Any, Dict, Optional

# MLflow tracking configurations - using single mlruns location
TRACKING_CONFIGS = {
    "local": {
        "tracking_uri": "file:./mlruns",
        "artifact_location": None,  # Use default
    },
    "server": {
        "tracking_uri": "http://localhost:5000",
        "artifact_location": None,  # Use server default
    },
}

# Default experiment names
EXPERIMENT_NAMES = {
    "default": "ML Experiments",
    "real_estate": "Real Estate Price Prediction",
    "hyperparameter_tuning": "Hyperparameter Tuning",
    "data_pipeline": "Real Estate Data Pipeline Experiments",
    "model_comparison": "Model Comparison Analysis",
    "cv_comparison": "Cross-Validation vs Holdout Comparison",
}

# Simplified MLflow configurations (no autolog to avoid conflicts)
SIMPLE_CONFIGS = {
    "minimal": {
        "enable_autolog": False,
    },
    "standard": {
        "enable_autolog": False,
    },
    "detailed": {
        "enable_autolog": False,
    },
}

# Standard tags for experiments
STANDARD_TAGS = {
    "framework": "sklearn",
    "problem_type": "regression",
    "domain": "real_estate",
    "data_source": "kc_house_data",
    "version": "1.0",
}

# Experiment-specific tag templates
EXPERIMENT_TAG_TEMPLATES = {
    "hyperparameter_tuning": {
        **STANDARD_TAGS,
        "experiment_type": "hyperparameter_tuning",
        "tuning_run": "true",
    },
    "data_pipeline": {
        **STANDARD_TAGS,
        "experiment_type": "data_pipeline_experiment",
    },
    "model_comparison": {
        **STANDARD_TAGS,
        "experiment_type": "model_comparison",
    },
    "cv_comparison": {
        **STANDARD_TAGS,
        "experiment_type": "cv_comparison",
        "validation_method": "cross_validation_vs_holdout",
    },
    "baseline": {
        **STANDARD_TAGS,
        "experiment_type": "baseline",
        "baseline_run": "true",
    },
}

# Metrics to track consistently across experiments
STANDARD_METRICS = [
    "train_mae",
    "train_mse",
    "train_rmse",
    "train_r2",
    "test_mae",
    "test_mse",
    "test_rmse",
    "test_r2",
    "overfitting_ratio_mae",
    "overfitting_ratio_r2",
]

# Parameters to log consistently
STANDARD_PARAMS = [
    "model_type",
    "test_size",
    "random_state",
    "dataset_size",
    "n_features",
    "scaler_type",
    "feature_engineering",
    "outlier_removal",
]

# Run naming patterns
RUN_NAME_PATTERNS = {
    "hyperparameter_tuning": "{model_type}_Tuning_{idx:02d}_n{n_estimators}_d{max_depth}_lr{learning_rate}",
    "data_pipeline": "Exp_{idx:03d}_{model_type}_{train_split}_{scaler}_{feature_eng}_outliers{outliers}",
    "model_comparison": "{model_type}_Comparison_{idx:02d}",
    "cv_comparison": "{model_type}_CV_vs_Holdout_{idx:02d}",
    "baseline": "{model_type}_Baseline",
}


def get_tracking_config(config_type: str = "local") -> Dict[str, Any]:
    """Get MLflow tracking configuration.

    Args:
        config_type: Type of tracking configuration

    Returns:
        Tracking configuration dictionary

    Raises:
        ValueError: If config_type is not supported
    """
    if config_type not in TRACKING_CONFIGS:
        raise ValueError(
            f"Unknown tracking config: {config_type}. Available: {list(TRACKING_CONFIGS.keys())}"
        )

    return TRACKING_CONFIGS[config_type].copy()


def get_experiment_name(experiment_type: str = "default") -> str:
    """Get experiment name for a given type.

    Args:
        experiment_type: Type of experiment

    Returns:
        Experiment name

    Raises:
        ValueError: If experiment_type is not supported
    """
    if experiment_type not in EXPERIMENT_NAMES:
        raise ValueError(
            f"Unknown experiment type: {experiment_type}. Available: {list(EXPERIMENT_NAMES.keys())}"
        )

    return EXPERIMENT_NAMES[experiment_type]


def get_simple_config(config_type: str = "minimal") -> Dict[str, Any]:
    """Get simplified configuration.

    Args:
        config_type: Type of configuration

    Returns:
        Configuration dictionary

    Raises:
        ValueError: If config_type is not supported
    """
    if config_type not in SIMPLE_CONFIGS:
        raise ValueError(
            f"Unknown config: {config_type}. Available: {list(SIMPLE_CONFIGS.keys())}"
        )

    return SIMPLE_CONFIGS[config_type].copy()


def get_experiment_tags(
    experiment_type: str, additional_tags: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """Get tags for an experiment.

    Args:
        experiment_type: Type of experiment
        additional_tags: Additional tags to include

    Returns:
        Dictionary of tags

    Raises:
        ValueError: If experiment_type is not supported
    """
    if experiment_type not in EXPERIMENT_TAG_TEMPLATES:
        # Use standard tags if no specific template
        tags = STANDARD_TAGS.copy()
        tags["experiment_type"] = experiment_type
    else:
        tags = EXPERIMENT_TAG_TEMPLATES[experiment_type].copy()

    if additional_tags:
        tags.update(additional_tags)

    return tags


def create_run_name(
    experiment_type: str,
    model_type: str,
    idx: int,
    params: Optional[Dict[str, Any]] = None,
) -> str:
    """Create a run name based on experiment type and parameters.

    Args:
        experiment_type: Type of experiment
        model_type: Type of model
        idx: Experiment index
        params: Additional parameters for name formatting

    Returns:
        Formatted run name
    """
    params = params or {}

    if experiment_type in RUN_NAME_PATTERNS:
        pattern = RUN_NAME_PATTERNS[experiment_type]
        try:
            return pattern.format(model_type=model_type.upper(), idx=idx, **params)
        except KeyError:
            # Fall back to simple naming if parameters are missing
            return f"{model_type.upper()}_{experiment_type}_{idx:02d}"
    else:
        # Default naming pattern
        return f"{model_type.upper()}_{experiment_type}_{idx:02d}"


def validate_mlflow_config(config: Dict[str, Any]) -> None:
    """Validate MLflow configuration.

    Args:
        config: Configuration to validate

    Raises:
        ValueError: If configuration is invalid
    """
    required_keys = ["tracking_uri", "experiment_name"]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required MLflow config key: {key}")

    # Validate tracking URI format
    tracking_uri = config["tracking_uri"]
    if not (
        tracking_uri.startswith("file:")
        or tracking_uri.startswith("http:")
        or tracking_uri.startswith("https:")
    ):
        raise ValueError(f"Invalid tracking URI format: {tracking_uri}")


class MLflowConfig:
    """MLflow configuration class for experiments."""

    def __init__(
        self,
        tracking_config: str = "local",
        experiment_type: str = "default",
        autolog_config: str = "minimal",
        enable_autolog: bool = True,
        additional_tags: Optional[Dict[str, str]] = None,
    ):
        """Initialize MLflow configuration.

        Args:
            tracking_config: Type of tracking configuration
            experiment_type: Type of experiment
            autolog_config: Type of autolog configuration
            enable_autolog: Whether to enable autologging
            additional_tags: Additional tags to include
        """
        self.tracking_config = get_tracking_config(tracking_config)
        self.experiment_name = get_experiment_name(experiment_type)
        self.experiment_type = experiment_type
        self.autolog_config = get_simple_config(autolog_config)
        self.enable_autolog = self.autolog_config.get("enable_autolog", False)
        self.tags = get_experiment_tags(experiment_type, additional_tags)

        # Validate configuration
        self._validate()

    def _validate(self) -> None:
        """Validate the configuration."""
        config_dict = {
            "tracking_uri": self.tracking_config["tracking_uri"],
            "experiment_name": self.experiment_name,
        }
        validate_mlflow_config(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        return {
            "tracking_uri": self.tracking_config["tracking_uri"],
            "experiment_name": self.experiment_name,
            "experiment_type": self.experiment_type,
            "autolog_config": self.autolog_config,
            "enable_autolog": self.enable_autolog,
            "tags": self.tags,
        }

    def get_setup_params(self) -> Dict[str, Any]:
        """Get parameters for MLflow setup.

        Returns:
            Setup parameters dictionary
        """
        return {
            "tracking_uri": self.tracking_config["tracking_uri"],
            "experiment_name": self.experiment_name,
            "enable_autolog": self.enable_autolog,
        }

    def create_run_name(
        self, model_type: str, idx: int, params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a run name for this configuration.

        Args:
            model_type: Type of model
            idx: Experiment index
            params: Additional parameters for name formatting

        Returns:
            Formatted run name
        """
        return create_run_name(self.experiment_type, model_type, idx, params)

    def get_tags_for_model(self, model_type: str) -> Dict[str, str]:
        """Get tags including model-specific information.

        Args:
            model_type: Type of model

        Returns:
            Dictionary of tags
        """
        tags = self.tags.copy()
        tags["model_type"] = model_type
        return tags


# Pre-configured MLflow setups for common use cases
COMMON_CONFIGS = {
    "local_tuning": MLflowConfig(
        tracking_config="local",
        experiment_type="hyperparameter_tuning",
        autolog_config="minimal",
    ),
    "local_comparison": MLflowConfig(
        tracking_config="local",
        experiment_type="model_comparison",
        autolog_config="standard",
    ),
    "local_data_pipeline": MLflowConfig(
        tracking_config="local",
        experiment_type="data_pipeline",
        autolog_config="minimal",
    ),
}


def get_common_config(config_name: str) -> MLflowConfig:
    """Get a pre-configured MLflow setup.

    Args:
        config_name: Name of the configuration

    Returns:
        MLflowConfig instance

    Raises:
        ValueError: If config_name is not supported
    """
    if config_name not in COMMON_CONFIGS:
        raise ValueError(
            f"Unknown common config: {config_name}. Available: {list(COMMON_CONFIGS.keys())}"
        )

    return COMMON_CONFIGS[config_name]
