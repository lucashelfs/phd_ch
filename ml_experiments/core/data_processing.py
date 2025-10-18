"""
Data processing utilities.

This module consolidates all data processing functionality that was duplicated
across multiple experiment scripts.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)


def load_sales_data(
    file_path: str, columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Load sales data from CSV file.

    Args:
        file_path: Path to the sales data CSV file
        columns: List of columns to select (if None, load all)

    Returns:
        DataFrame with sales data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded sales data with shape {df.shape}")

        if columns:
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in sales data: {missing_cols}")
            df = df[columns]
            logger.info(f"Selected columns, new shape: {df.shape}")

        return df
    except FileNotFoundError:
        logger.error(f"Sales data file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading sales data: {e}")
        raise


def load_demographics_data(file_path: str) -> pd.DataFrame:
    """Load demographics data from CSV file.

    Args:
        file_path: Path to the demographics data CSV file

    Returns:
        DataFrame with demographics data

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded demographics data with shape {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Demographics data file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading demographics data: {e}")
        raise


def merge_data(sales_df: pd.DataFrame, demographics_df: pd.DataFrame) -> pd.DataFrame:
    """Merge sales and demographics data on zipcode.

    Args:
        sales_df: Sales DataFrame
        demographics_df: Demographics DataFrame

    Returns:
        Merged DataFrame with zipcode removed (matches original create_model.py logic)

    Raises:
        ValueError: If zipcode column is missing
    """
    if "zipcode" not in sales_df.columns:
        raise ValueError("Sales data must contain 'zipcode' column")

    if "zipcode" not in demographics_df.columns:
        raise ValueError("Demographics data must contain 'zipcode' column")

    # Merge on zipcode
    merged_df = sales_df.merge(demographics_df, on="zipcode", how="left")
    logger.info(f"Merged data shape: {merged_df.shape}")

    # Check for missing demographics data
    missing_demographics = (
        merged_df.isnull().sum().sum() - sales_df.isnull().sum().sum()
    )
    if missing_demographics > 0:
        logger.warning(f"Found {missing_demographics} missing values after merge")

    # Drop zipcode after merge (matches original create_model.py preprocessing)
    # This ensures the model trains without zipcode as a feature, matching API expectations
    merged_df = merged_df.drop(columns="zipcode")
    logger.info(f"Dropped zipcode column, final shape: {merged_df.shape}")

    return merged_df


def remove_outliers_iqr(
    df: pd.DataFrame, target_column: str, multiplier: float = 1.5
) -> pd.DataFrame:
    """Remove outliers using IQR method.

    Args:
        df: Input DataFrame
        target_column: Name of target column
        multiplier: IQR multiplier for outlier detection

    Returns:
        DataFrame with outliers removed
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")

    Q1 = df[target_column].quantile(0.25)
    Q3 = df[target_column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    initial_count = len(df)
    df_clean = df[
        (df[target_column] >= lower_bound) & (df[target_column] <= upper_bound)
    ]
    final_count = len(df_clean)

    removed_count = initial_count - final_count
    logger.info(
        f"Removed {removed_count} outliers ({removed_count / initial_count * 100:.1f}%)"
    )

    return df_clean


def apply_log_transform(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Apply log transformation to specified columns.

    Args:
        df: Input DataFrame
        columns: List of columns to transform

    Returns:
        DataFrame with log-transformed columns
    """
    df_transformed = df.copy()

    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found, skipping log transform")
            continue

        # Add small constant to avoid log(0)
        df_transformed[col] = np.log1p(df_transformed[col])
        logger.info(f"Applied log transform to column: {col}")

    return df_transformed


def apply_polynomial_features(
    X: pd.DataFrame, degree: int = 2, interaction_only: bool = True
) -> pd.DataFrame:
    """Apply polynomial feature engineering.

    Args:
        X: Input features DataFrame
        degree: Polynomial degree
        interaction_only: Whether to include only interaction terms

    Returns:
        DataFrame with polynomial features
    """
    poly = PolynomialFeatures(
        degree=degree, interaction_only=interaction_only, include_bias=False
    )
    X_poly = poly.fit_transform(X)

    # Create feature names
    feature_names = poly.get_feature_names_out(X.columns)

    # Convert back to DataFrame
    X_poly_df = pd.DataFrame(X_poly, columns=feature_names, index=X.index)

    logger.info(
        f"Created polynomial features: {X.shape[1]} -> {X_poly_df.shape[1]} features"
    )

    return X_poly_df


def split_features_target(
    df: pd.DataFrame, target_column: str = "price"
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into features and target.

    Args:
        df: Input DataFrame
        target_column: Name of target column

    Returns:
        Tuple of (features DataFrame, target Series)

    Raises:
        ValueError: If target column is missing
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    logger.info(f"Split data: {X.shape[1]} features, {len(y)} samples")

    return X, y


def train_test_split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets.

    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Proportion of test set
        random_state: Random state for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test


def load_and_prepare_data(
    sales_path: str,
    demographics_path: str,
    sales_columns: Optional[List[str]] = None,
    target_column: str = "price",
    test_size: float = 0.2,
    random_state: int = 42,
    remove_outliers: bool = False,
    outlier_multiplier: float = 1.5,
    feature_engineering: str = "none",
    log_transform_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
    """Load and prepare data for modeling.

    Args:
        sales_path: Path to sales data
        demographics_path: Path to demographics data
        sales_columns: Columns to select from sales data
        target_column: Name of target column
        test_size: Test set proportion
        random_state: Random state
        remove_outliers: Whether to remove outliers
        outlier_multiplier: IQR multiplier for outlier removal
        feature_engineering: Type of feature engineering ('none', 'poly2', 'log_transform')
        log_transform_columns: Columns to log transform

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, metadata)
    """
    metadata = {}

    # Load data
    sales_df = load_sales_data(sales_path, sales_columns)
    demographics_df = load_demographics_data(demographics_path)

    # Merge data
    df = merge_data(sales_df, demographics_df)
    metadata["initial_shape"] = df.shape

    # Remove outliers if requested
    if remove_outliers:
        df = remove_outliers_iqr(df, target_column, outlier_multiplier)
        metadata["shape_after_outlier_removal"] = df.shape

    # Split features and target
    X, y = split_features_target(df, target_column)

    # Apply feature engineering
    if feature_engineering == "poly2":
        X = apply_polynomial_features(X, degree=2, interaction_only=True)
        metadata["feature_engineering"] = "polynomial_degree_2"
    elif feature_engineering == "log_transform" and log_transform_columns:
        X = apply_log_transform(X, log_transform_columns)
        metadata["feature_engineering"] = "log_transform"
        metadata["log_transform_columns"] = log_transform_columns
    else:
        metadata["feature_engineering"] = "none"

    metadata["final_features"] = X.shape[1]
    metadata["final_samples"] = len(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, test_size=test_size, random_state=random_state
    )

    metadata["train_samples"] = len(X_train)
    metadata["test_samples"] = len(X_test)

    return X_train, X_test, y_train, y_test, metadata


class DataProcessor:
    """Data processing pipeline class."""

    def __init__(
        self,
        sales_path: str,
        demographics_path: str,
        sales_columns: Optional[List[str]] = None,
        target_column: str = "price",
        remove_outliers: bool = False,
        outlier_multiplier: float = 1.5,
        feature_engineering: str = "none",
        log_transform_columns: Optional[List[str]] = None,
    ):
        """Initialize data processor.

        Args:
            sales_path: Path to sales data
            demographics_path: Path to demographics data
            sales_columns: Columns to select from sales data
            target_column: Name of target column
            remove_outliers: Whether to remove outliers
            outlier_multiplier: IQR multiplier for outlier removal
            feature_engineering: Type of feature engineering
            log_transform_columns: Columns to log transform
        """
        self.sales_path = sales_path
        self.demographics_path = demographics_path
        self.sales_columns = sales_columns
        self.target_column = target_column
        self.remove_outliers = remove_outliers
        self.outlier_multiplier = outlier_multiplier
        self.feature_engineering = feature_engineering
        self.log_transform_columns = log_transform_columns or []

        self.data: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}

    def load_data(self) -> pd.DataFrame:
        """Load and merge data.

        Returns:
            Merged DataFrame
        """
        sales_df = load_sales_data(self.sales_path, self.sales_columns)
        demographics_df = load_demographics_data(self.demographics_path)
        self.data = merge_data(sales_df, demographics_df)
        self.metadata["initial_shape"] = self.data.shape
        return self.data

    def preprocess(self) -> pd.DataFrame:
        """Apply preprocessing steps.

        Returns:
            Preprocessed DataFrame
        """
        if self.data is None:
            self.load_data()

        # Remove outliers if requested
        if self.remove_outliers and self.data is not None:
            self.data = remove_outliers_iqr(
                self.data, self.target_column, self.outlier_multiplier
            )
            self.metadata["shape_after_outlier_removal"] = self.data.shape

        # This should never be None at this point, but add check for type safety
        if self.data is None:
            raise RuntimeError("Data is None after preprocessing")

        return self.data

    def engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering.

        Args:
            X: Features DataFrame

        Returns:
            Engineered features DataFrame
        """
        if self.feature_engineering == "poly2":
            X = apply_polynomial_features(X, degree=2, interaction_only=True)
            self.metadata["feature_engineering"] = "polynomial_degree_2"
        elif self.feature_engineering == "log_transform" and self.log_transform_columns:
            X = apply_log_transform(X, self.log_transform_columns)
            self.metadata["feature_engineering"] = "log_transform"
            self.metadata["log_transform_columns"] = self.log_transform_columns
        else:
            self.metadata["feature_engineering"] = "none"

        return X

    def prepare_for_modeling(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare data for modeling.

        Args:
            test_size: Test set proportion
            random_state: Random state

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Preprocess data
        self.preprocess()

        # Split features and target - data is guaranteed to be non-None after preprocess()
        if self.data is None:
            raise RuntimeError("Data is None after preprocessing")
        X, y = split_features_target(self.data, self.target_column)

        # Apply feature engineering
        X = self.engineer_features(X)

        self.metadata["final_features"] = X.shape[1]
        self.metadata["final_samples"] = len(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split_data(
            X, y, test_size=test_size, random_state=random_state
        )

        self.metadata["train_samples"] = len(X_train)
        self.metadata["test_samples"] = len(X_test)

        return X_train, X_test, y_train, y_test

    def get_metadata(self) -> Dict[str, Any]:
        """Get processing metadata.

        Returns:
            Metadata dictionary
        """
        return self.metadata.copy()
