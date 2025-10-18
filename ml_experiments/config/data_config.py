"""
Data configuration settings.

This module centralizes all data-related configuration parameters
that were scattered across multiple experiment scripts.
"""

from typing import Any, Dict

from sklearn.preprocessing import PolynomialFeatures

# Data file paths
SALES_PATH = "data/kc_house_data.csv"
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"

# Alternative paths for experiments directory
EXPERIMENTS_SALES_PATH = "../../data/kc_house_data.csv"
EXPERIMENTS_DEMOGRAPHICS_PATH = "../../data/zipcode_demographics.csv"

# Column selection from sales data
SALES_COLUMN_SELECTION = [
    "price",
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "sqft_above",
    "sqft_basement",
    "zipcode",
]

# Data preprocessing configurations
SCALER_CONFIGS = {
    "robust": {
        "quantile_range": (25.0, 75.0),
        "with_centering": True,
        "with_scaling": True,
    },
    "standard": {
        "with_mean": True,
        "with_std": True,
    },
    "minmax": {
        "feature_range": (0, 1),
    },
}

# Feature engineering configurations
FEATURE_ENGINEERING_CONFIGS = {
    "none": None,
    "poly2": {
        "degree": 2,
        "interaction_only": True,
        "include_bias": False,
    },
    "log_transform": "log_transform",  # Special marker for log transform
}

# Data split configurations
TRAIN_TEST_SPLITS = {
    "default": {"test_size": 0.2, "random_state": 42},
    "small_test": {"test_size": 0.15, "random_state": 42},
    "large_test": {"test_size": 0.3, "random_state": 42},
}

# Experiment configurations for data pipeline experiments
DATA_PIPELINE_EXPERIMENT_CONFIGS = {
    "test_sizes": [0.20, 0.25, 0.30],  # 80/20, 75/25, 70/30 splits
    "scalers": {
        "robust": SCALER_CONFIGS["robust"],
        "standard": SCALER_CONFIGS["standard"],
        "minmax": SCALER_CONFIGS["minmax"],
    },
    "feature_engineering": FEATURE_ENGINEERING_CONFIGS,
    "outlier_removal": [False, True],
}

# Outlier removal configurations
OUTLIER_CONFIGS = {
    "iqr": {
        "method": "iqr",
        "multiplier": 1.5,
    },
    "iqr_strict": {
        "method": "iqr",
        "multiplier": 1.0,
    },
    "iqr_loose": {
        "method": "iqr",
        "multiplier": 2.0,
    },
}

# Feature selection configurations
FEATURE_SELECTION_CONFIGS = {
    "kbest_10": {"k": 10, "score_func": "f_regression"},
    "kbest_20": {"k": 20, "score_func": "f_regression"},
    "kbest_50": {"k": 50, "score_func": "f_regression"},
}

# Data validation thresholds
DATA_VALIDATION_THRESHOLDS = {
    "min_samples": 100,
    "max_missing_ratio": 0.5,  # 50% missing values threshold
    "min_target_variance": 1e-6,
}


def get_data_paths(use_experiments_path: bool = False) -> Dict[str, str]:
    """Get data file paths based on context.

    Args:
        use_experiments_path: Whether to use paths relative to experiments directory

    Returns:
        Dictionary with sales and demographics paths
    """
    if use_experiments_path:
        return {
            "sales_path": EXPERIMENTS_SALES_PATH,
            "demographics_path": EXPERIMENTS_DEMOGRAPHICS_PATH,
        }
    else:
        return {
            "sales_path": SALES_PATH,
            "demographics_path": DEMOGRAPHICS_PATH,
        }


def get_scaler_config(scaler_type: str) -> Dict[str, Any]:
    """Get scaler configuration.

    Args:
        scaler_type: Type of scaler

    Returns:
        Scaler configuration dictionary
    """
    if scaler_type not in SCALER_CONFIGS:
        raise ValueError(
            f"Unknown scaler type: {scaler_type}. Available: {list(SCALER_CONFIGS.keys())}"
        )

    return SCALER_CONFIGS[scaler_type].copy()


def get_feature_engineering_config(feature_eng_type: str) -> Any:
    """Get feature engineering configuration.

    Args:
        feature_eng_type: Type of feature engineering

    Returns:
        Feature engineering configuration
    """
    if feature_eng_type not in FEATURE_ENGINEERING_CONFIGS:
        raise ValueError(
            f"Unknown feature engineering type: {feature_eng_type}. Available: {list(FEATURE_ENGINEERING_CONFIGS.keys())}"
        )

    config = FEATURE_ENGINEERING_CONFIGS[feature_eng_type]

    # Return PolynomialFeatures instance for poly2
    if feature_eng_type == "poly2" and isinstance(config, dict):
        return PolynomialFeatures(**config)

    return config


def get_train_test_split_config(split_type: str = "default") -> Dict[str, Any]:
    """Get train-test split configuration.

    Args:
        split_type: Type of split configuration

    Returns:
        Split configuration dictionary
    """
    if split_type not in TRAIN_TEST_SPLITS:
        raise ValueError(
            f"Unknown split type: {split_type}. Available: {list(TRAIN_TEST_SPLITS.keys())}"
        )

    return TRAIN_TEST_SPLITS[split_type].copy()


def get_outlier_config(outlier_type: str = "iqr") -> Dict[str, Any]:
    """Get outlier removal configuration.

    Args:
        outlier_type: Type of outlier removal

    Returns:
        Outlier configuration dictionary
    """
    if outlier_type not in OUTLIER_CONFIGS:
        raise ValueError(
            f"Unknown outlier type: {outlier_type}. Available: {list(OUTLIER_CONFIGS.keys())}"
        )

    return OUTLIER_CONFIGS[outlier_type].copy()


def validate_data_config(config: Dict[str, Any]) -> None:
    """Validate data configuration parameters.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid
    """
    required_keys = ["sales_path", "demographics_path", "sales_columns"]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")

    # Validate test size
    if "test_size" in config:
        test_size = config["test_size"]
        if not (0 < test_size < 1):
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

    # Validate scaler type
    if "scaler_type" in config:
        scaler_type = config["scaler_type"]
        if scaler_type not in SCALER_CONFIGS:
            raise ValueError(
                f"Invalid scaler_type: {scaler_type}. Available: {list(SCALER_CONFIGS.keys())}"
            )


class DataConfig:
    """Data configuration class for experiments."""

    def __init__(
        self,
        use_experiments_path: bool = False,
        scaler_type: str = "robust",
        feature_engineering: str = "none",
        test_size: float = 0.2,
        remove_outliers: bool = False,
        outlier_type: str = "iqr",
        random_state: int = 42,
    ):
        """Initialize data configuration.

        Args:
            use_experiments_path: Whether to use experiments directory paths
            scaler_type: Type of scaler to use
            feature_engineering: Type of feature engineering
            test_size: Test set size ratio
            remove_outliers: Whether to remove outliers
            outlier_type: Type of outlier removal
            random_state: Random state for reproducibility
        """
        self.paths = get_data_paths(use_experiments_path)
        self.sales_columns = SALES_COLUMN_SELECTION.copy()
        self.scaler_type = scaler_type
        self.scaler_config = get_scaler_config(scaler_type)
        self.feature_engineering = feature_engineering
        self.feature_eng_config = get_feature_engineering_config(feature_engineering)
        self.test_size = test_size
        self.remove_outliers = remove_outliers
        self.outlier_config = (
            get_outlier_config(outlier_type) if remove_outliers else None
        )
        self.random_state = random_state

        # Validate configuration
        self._validate()

    def _validate(self) -> None:
        """Validate the configuration."""
        config_dict = {
            "sales_path": self.paths["sales_path"],
            "demographics_path": self.paths["demographics_path"],
            "sales_columns": self.sales_columns,
            "test_size": self.test_size,
            "scaler_type": self.scaler_type,
        }
        validate_data_config(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        return {
            "sales_path": self.paths["sales_path"],
            "demographics_path": self.paths["demographics_path"],
            "sales_columns": self.sales_columns,
            "scaler_type": self.scaler_type,
            "scaler_config": self.scaler_config,
            "feature_engineering": self.feature_engineering,
            "test_size": self.test_size,
            "remove_outliers": self.remove_outliers,
            "outlier_config": self.outlier_config,
            "random_state": self.random_state,
        }

    def get_processing_params(self) -> Dict[str, Any]:
        """Get parameters for data processing.

        Returns:
            Processing parameters dictionary
        """
        return {
            "remove_outliers_flag": self.remove_outliers,
            "outlier_method": self.outlier_config["method"]
            if self.outlier_config
            else None,
            "feature_engineering": self.feature_engineering,
            "feature_eng_config": self.feature_eng_config
            if isinstance(self.feature_eng_config, dict)
            else None,
        }
