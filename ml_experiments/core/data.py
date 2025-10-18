"""
Data loading and preprocessing utilities.

This module consolidates all data-related functionality that was duplicated
across multiple experiment scripts.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


def load_data(
    sales_path: str, demographics_path: str, sales_column_selection: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the target and feature data by merging sales and demographics.

    Args:
        sales_path: Path to CSV file with home sale data
        demographics_path: Path to CSV file with demographics data
        sales_column_selection: List of columns from sales data to be used as features

    Returns:
        Tuple containing two elements: a DataFrame and a Series of the same
        length. The DataFrame contains features for machine learning, the
        series contains the target variable (home sale price).
    """
    data = pd.read_csv(
        sales_path, usecols=sales_column_selection, dtype={"zipcode": str}
    )
    demographics = pd.read_csv(demographics_path, dtype={"zipcode": str})

    merged_data = data.merge(demographics, how="left", on="zipcode").drop(
        columns="zipcode"
    )
    # Remove the target variable from the dataframe, features will remain
    y = merged_data.pop("price")
    x = merged_data

    return x, y


def remove_outliers(
    x: pd.DataFrame, y: pd.Series, method: str = "iqr", iqr_multiplier: float = 1.5
) -> Tuple[pd.DataFrame, pd.Series]:
    """Remove outliers using specified method.

    Args:
        x: Feature DataFrame
        y: Target Series
        method: Method for outlier detection ("iqr" supported)
        iqr_multiplier: Multiplier for IQR-based outlier detection

    Returns:
        Tuple of cleaned features and target
    """
    if method == "iqr":
        # Remove outliers based on target variable
        Q1 = y.quantile(0.25)
        Q3 = y.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR

        mask = (y >= lower_bound) & (y <= upper_bound)
        return x[mask].reset_index(drop=True), y[mask].reset_index(drop=True)

    return x, y


def apply_feature_engineering(
    x: pd.DataFrame, feature_eng_type: str, feature_eng_config: Optional[dict] = None
) -> pd.DataFrame:
    """Apply feature engineering transformations.

    Args:
        x: Input features DataFrame
        feature_eng_type: Type of feature engineering to apply
        feature_eng_config: Configuration for feature engineering

    Returns:
        Transformed features DataFrame
    """
    if feature_eng_type == "log_transform":
        # Apply log transform to skewed numerical features
        x_transformed = x.copy()
        numerical_cols = x.select_dtypes(include=[np.number]).columns

        for col in numerical_cols:
            if (x[col] > 0).all():  # Only apply log to positive values
                x_transformed[f"{col}_log"] = np.log1p(x[col])

        return x_transformed

    elif feature_eng_type == "poly2":
        # Apply polynomial features (interactions only to avoid explosion)
        poly_config = feature_eng_config or {
            "degree": 2,
            "interaction_only": True,
            "include_bias": False,
        }
        poly_features = PolynomialFeatures(**poly_config)
        x_poly = poly_features.fit_transform(x)
        feature_names = poly_features.get_feature_names_out(x.columns)
        return pd.DataFrame(x_poly, columns=feature_names, index=x.index)

    elif feature_eng_type == "none":
        return x

    else:
        raise ValueError(f"Unknown feature engineering type: {feature_eng_type}")


def get_data_info(x: pd.DataFrame, y: pd.Series) -> dict:
    """Get dataset information for logging.

    Args:
        x: Feature DataFrame
        y: Target Series

    Returns:
        Dictionary with dataset statistics
    """
    return {
        "dataset_size": len(x),
        "n_features": len(x.columns),
        "target_mean": float(y.mean()),
        "target_std": float(y.std()),
        "target_min": float(y.min()),
        "target_max": float(y.max()),
        "feature_names": list(x.columns),
        "missing_values": int(x.isnull().sum().sum()),
        "target_missing": int(y.isnull().sum()),
    }


def validate_data(x: pd.DataFrame, y: pd.Series) -> None:
    """Validate input data for common issues.

    Args:
        x: Feature DataFrame
        y: Target Series

    Raises:
        ValueError: If data validation fails
    """
    if len(x) != len(y):
        raise ValueError(f"Feature and target length mismatch: {len(x)} vs {len(y)}")

    if len(x) == 0:
        raise ValueError("Empty dataset provided")

    if x.isnull().all().any():
        null_cols = x.columns[x.isnull().all()].tolist()
        raise ValueError(f"Columns with all null values: {null_cols}")

    if y.isnull().all():
        raise ValueError("Target variable has all null values")

    # Check for infinite values
    if np.isinf(x.select_dtypes(include=[np.number])).any().any():
        raise ValueError("Infinite values found in features")

    if np.isinf(y).any():
        raise ValueError("Infinite values found in target")


class DataProcessor:
    """Encapsulates data processing pipeline for experiments."""

    def __init__(
        self,
        sales_path: str,
        demographics_path: str,
        sales_columns: List[str],
        remove_outliers_flag: bool = False,
        outlier_method: str = "iqr",
        feature_engineering: str = "none",
        feature_eng_config: Optional[dict] = None,
    ):
        """Initialize data processor.

        Args:
            sales_path: Path to sales data CSV
            demographics_path: Path to demographics data CSV
            sales_columns: Columns to select from sales data
            remove_outliers_flag: Whether to remove outliers
            outlier_method: Method for outlier removal
            feature_engineering: Type of feature engineering
            feature_eng_config: Configuration for feature engineering
        """
        self.sales_path = sales_path
        self.demographics_path = demographics_path
        self.sales_columns = sales_columns
        self.remove_outliers_flag = remove_outliers_flag
        self.outlier_method = outlier_method
        self.feature_engineering = feature_engineering
        self.feature_eng_config = feature_eng_config

        self._original_shape = None
        self._processed_shape = None

    def load_and_process(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and process data according to configuration.

        Returns:
            Tuple of processed features and target
        """
        # Load raw data
        x, y = load_data(self.sales_path, self.demographics_path, self.sales_columns)
        self._original_shape = x.shape

        # Validate data
        validate_data(x, y)

        # Remove outliers if specified
        if self.remove_outliers_flag:
            x, y = remove_outliers(x, y, method=self.outlier_method)

        # Apply feature engineering
        if self.feature_engineering != "none":
            x = apply_feature_engineering(
                x, self.feature_engineering, self.feature_eng_config
            )

        self._processed_shape = x.shape

        # Final validation
        validate_data(x, y)

        return x, y

    def get_processing_info(self) -> dict:
        """Get information about data processing steps.

        Returns:
            Dictionary with processing information
        """
        return {
            "original_shape": self._original_shape,
            "processed_shape": self._processed_shape,
            "outliers_removed": self.remove_outliers_flag,
            "outlier_method": self.outlier_method
            if self.remove_outliers_flag
            else None,
            "feature_engineering": self.feature_engineering,
            "samples_removed": (
                self._original_shape[0] - self._processed_shape[0]
                if self._original_shape and self._processed_shape
                else 0
            ),
            "features_added": (
                self._processed_shape[1] - self._original_shape[1]
                if self._original_shape and self._processed_shape
                else 0
            ),
        }
