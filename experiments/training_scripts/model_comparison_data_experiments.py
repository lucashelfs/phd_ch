import json
import pathlib
import pickle
from typing import List, Tuple, Dict, Any
from itertools import product

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from sklearn import model_selection
from sklearn import neighbors
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature
import xgboost as xgb
import lightgbm as lgb

# Configuration
SALES_PATH = "../../data/kc_house_data.csv"
DEMOGRAPHICS_PATH = "../../data/zipcode_demographics.csv"
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
OUTPUT_DIR = "model"

# MLflow Configuration
EXPERIMENT_NAME = "Real Estate Data Pipeline Experiments"
TRACKING_URI = "file:../../mlruns"

# Best parameters from hyperparameter tuning
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

KNN_PARAMS = {
    "n_neighbors": 5,
    "weights": "uniform",
    "algorithm": "auto",
    "leaf_size": 30,
    "p": 2,
    "metric": "minkowski",
}

# Experiment configurations
EXPERIMENT_CONFIGS = {
    "test_sizes": [0.20, 0.25, 0.30],  # 80/20, 75/25, 70/30 splits
    "scalers": {
        "robust": preprocessing.RobustScaler(quantile_range=(25.0, 75.0)),
        "standard": preprocessing.StandardScaler(),
        "minmax": preprocessing.MinMaxScaler(),
    },
    "feature_engineering": {
        "none": None,
        "poly2": PolynomialFeatures(
            degree=2, interaction_only=True, include_bias=False
        ),
        "log_transform": "log_transform",
    },
    "outlier_removal": [False, True],
}


def load_data(
    sales_path: str, demographics_path: str, sales_column_selection: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the target and feature data by merging sales and demographics."""
    data = pd.read_csv(
        sales_path, usecols=sales_column_selection, dtype={"zipcode": str}
    )
    demographics = pd.read_csv(demographics_path, dtype={"zipcode": str})

    merged_data = data.merge(demographics, how="left", on="zipcode").drop(
        columns="zipcode"
    )
    y = merged_data.pop("price")
    x = merged_data

    return x, y


def remove_outliers(
    x: pd.DataFrame, y: pd.Series, method: str = "iqr"
) -> Tuple[pd.DataFrame, pd.Series]:
    """Remove outliers using IQR method."""
    if method == "iqr":
        # Remove outliers based on target variable
        Q1 = y.quantile(0.25)
        Q3 = y.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        mask = (y >= lower_bound) & (y <= upper_bound)
        return x[mask].reset_index(drop=True), y[mask].reset_index(drop=True)

    return x, y


def apply_feature_engineering(x: pd.DataFrame, feature_eng_type: str) -> pd.DataFrame:
    """Apply feature engineering transformations."""
    if feature_eng_type == "log_transform":
        # Apply log transform to skewed numerical features
        x_transformed = x.copy()
        numerical_cols = x.select_dtypes(include=[np.number]).columns

        for col in numerical_cols:
            if (x[col] > 0).all():  # Only apply log to positive values
                x_transformed[f"{col}_log"] = np.log1p(x[col])

        return x_transformed

    elif isinstance(feature_eng_type, PolynomialFeatures):
        # Apply polynomial features (interactions only to avoid explosion)
        x_poly = feature_eng_type.fit_transform(x)
        feature_names = feature_eng_type.get_feature_names_out(x.columns)
        return pd.DataFrame(x_poly, columns=feature_names, index=x.index)

    return x


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Calculate regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


def create_model_pipeline(model_type: str, scaler, feature_selector=None):
    """Create a model pipeline with preprocessing."""
    steps = []

    # Add scaler
    steps.append(("scaler", scaler))

    # Add feature selector if specified
    if feature_selector is not None:
        steps.append(("feature_selector", feature_selector))

    # Add model
    if model_type == "knn":
        steps.append(("model", neighbors.KNeighborsRegressor(**KNN_PARAMS)))
    elif model_type == "xgboost":
        steps.append(("model", xgb.XGBRegressor(**BEST_XGBOOST_PARAMS)))
    elif model_type == "lightgbm":
        steps.append(("model", lgb.LGBMRegressor(**BEST_LIGHTGBM_PARAMS)))

    return pipeline.Pipeline(steps)


def run_experiment(
    x: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    test_size: float,
    scaler_name: str,
    scaler,
    feature_eng_type: str,
    remove_outliers_flag: bool,
    experiment_idx: int,
    total_experiments: int,
) -> Dict[str, Any]:
    """Run a single experiment configuration."""

    # Create experiment identifier
    exp_id = f"{model_type}_{int((1 - test_size) * 100)}_{scaler_name}_{feature_eng_type}_outliers{remove_outliers_flag}"
    run_name = f"Exp_{experiment_idx:03d}_{exp_id}"

    with mlflow.start_run(run_name=run_name) as run:
        print(f"[{experiment_idx}/{total_experiments}] Running: {run_name}")

        # Prepare data
        x_processed = x.copy()
        y_processed = y.copy()

        # Remove outliers if specified
        if remove_outliers_flag:
            x_processed, y_processed = remove_outliers(x_processed, y_processed)
            print(f"   Outliers removed: {len(x)} -> {len(x_processed)} samples")

        # Apply feature engineering
        if feature_eng_type != "none":
            x_processed = apply_feature_engineering(
                x_processed, EXPERIMENT_CONFIGS["feature_engineering"][feature_eng_type]
            )
            print(f"   Features after engineering: {x_processed.shape[1]}")

        # Feature selection for polynomial features (to avoid curse of dimensionality)
        feature_selector = None
        if feature_eng_type == "poly2" and x_processed.shape[1] > 50:
            feature_selector = SelectKBest(score_func=f_regression, k=50)

        # Split data
        x_train, x_test, y_train, y_test = model_selection.train_test_split(
            x_processed, y_processed, test_size=test_size, random_state=42
        )

        # Create and train model
        model = create_model_pipeline(model_type, scaler, feature_selector)
        model.fit(x_train, y_train)

        # Make predictions
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        # Calculate metrics
        train_metrics = calculate_metrics(y_train, y_train_pred)
        test_metrics = calculate_metrics(y_test, y_test_pred)

        print(f"   Test R²: {test_metrics['r2']:.4f}, MAE: ${test_metrics['mae']:,.0f}")

        # Log parameters
        mlflow.log_params(
            {
                "model_type": model_type,
                "test_size": test_size,
                "train_size": 1 - test_size,
                "scaler_type": scaler_name,
                "feature_engineering": feature_eng_type,
                "outlier_removal": remove_outliers_flag,
                "original_features": len(x.columns),
                "processed_features": x_processed.shape[1],
                "final_features": x_train.shape[1]
                if feature_selector is None
                else feature_selector.k,
                "training_samples": len(x_train),
                "test_samples": len(x_test),
                "outliers_removed": len(x) - len(x_processed)
                if remove_outliers_flag
                else 0,
            }
        )

        # Log metrics
        for metric_name, value in train_metrics.items():
            mlflow.log_metric(f"train_{metric_name}", value)

        for metric_name, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)

        # Log additional metrics
        mlflow.log_metric(
            "overfitting_ratio_mae", test_metrics["mae"] / train_metrics["mae"]
        )
        mlflow.log_metric(
            "overfitting_ratio_r2", train_metrics["r2"] / test_metrics["r2"]
        )

        # Set tags
        mlflow.set_tags(
            {
                "experiment_type": "data_pipeline_experiment",
                "model_family": model_type,
                "preprocessing": f"{scaler_name}_{feature_eng_type}",
                "data_split": f"{int((1 - test_size) * 100)}-{int(test_size * 100)}",
                "outlier_handling": "removed" if remove_outliers_flag else "kept",
            }
        )

        return {
            "run_id": run.info.run_id,
            "experiment_config": {
                "model_type": model_type,
                "test_size": test_size,
                "scaler": scaler_name,
                "feature_eng": feature_eng_type,
                "outliers_removed": remove_outliers_flag,
            },
            "test_r2": test_metrics["r2"],
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "overfitting_ratio_r2": train_metrics["r2"] / test_metrics["r2"],
            "final_features": x_train.shape[1]
            if feature_selector is None
            else feature_selector.k,
        }


def main():
    """Run comprehensive data pipeline experiments."""

    # Set MLflow tracking
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Real Estate Data Pipeline Experiments")
    print("=" * 60)

    # Load data
    print(f"Loading data from {SALES_PATH} and {DEMOGRAPHICS_PATH}")
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
    print(f"Dataset loaded: {len(x)} samples, {len(x.columns)} features")

    # Generate all experiment combinations
    model_types = ["knn", "xgboost", "lightgbm"]

    experiments = []
    for model_type in model_types:
        for test_size in EXPERIMENT_CONFIGS["test_sizes"]:
            for scaler_name, scaler in EXPERIMENT_CONFIGS["scalers"].items():
                for feature_eng_type in EXPERIMENT_CONFIGS[
                    "feature_engineering"
                ].keys():
                    for outlier_removal in EXPERIMENT_CONFIGS["outlier_removal"]:
                        experiments.append(
                            {
                                "model_type": model_type,
                                "test_size": test_size,
                                "scaler_name": scaler_name,
                                "scaler": scaler,
                                "feature_eng_type": feature_eng_type,
                                "outlier_removal": outlier_removal,
                            }
                        )

    total_experiments = len(experiments)
    print(f"\nRunning {total_experiments} experiments")
    print("=" * 60)

    # Run experiments
    results = []
    for idx, exp_config in enumerate(experiments, 1):
        try:
            result = run_experiment(
                x,
                y,
                exp_config["model_type"],
                exp_config["test_size"],
                exp_config["scaler_name"],
                exp_config["scaler"],
                exp_config["feature_eng_type"],
                exp_config["outlier_removal"],
                idx,
                total_experiments,
            )
            results.append(result)

        except Exception as e:
            print(f"   Error in experiment {idx}: {str(e)}")
            continue

    # Analyze results
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)

    if results:
        # Sort by test R²
        results_sorted = sorted(results, key=lambda x: x["test_r2"], reverse=True)

        print(f"\nTop 10 Configurations (by Test R²):")
        print("-" * 60)

        for i, result in enumerate(results_sorted[:10]):
            config = result["experiment_config"]
            print(
                f"\n{i + 1}. R²: {result['test_r2']:.4f} | MAE: ${result['test_mae']:,.0f}"
            )
            print(
                f"   Model: {config['model_type']}, Split: {int((1 - config['test_size']) * 100)}/{int(config['test_size'] * 100)}"
            )
            print(f"   Preprocessing: {config['scaler']} + {config['feature_eng']}")
            print(f"   Outliers: {'Removed' if config['outliers_removed'] else 'Kept'}")
            print(f"   Features: {result['final_features']}")

        # Best model summary
        best_result = results_sorted[0]
        print(f"\nBEST CONFIGURATION:")
        print(f"   Test R²: {best_result['test_r2']:.4f}")
        print(f"   Test MAE: ${best_result['test_mae']:,.2f}")
        print(f"   Test RMSE: ${best_result['test_rmse']:,.2f}")
        print(f"   Overfitting Ratio: {best_result['overfitting_ratio_r2']:.4f}")
        print(f"   Configuration: {best_result['experiment_config']}")

        # Save summary
        summary = {
            "experiment_type": "Data Pipeline Experiments",
            "total_experiments": len(results),
            "best_result": best_result,
            "top_10_results": results_sorted[:10],
            "experiment_configs": {
                "test_sizes": EXPERIMENT_CONFIGS["test_sizes"],
                "scalers": list(EXPERIMENT_CONFIGS["scalers"].keys()),
                "feature_engineering": list(
                    EXPERIMENT_CONFIGS["feature_engineering"].keys()
                ),
                "outlier_removal": EXPERIMENT_CONFIGS["outlier_removal"],
            },
        }

        with open("data_pipeline_experiments_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\nSummary saved to: data_pipeline_experiments_summary.json")

    else:
        print("No successful experiments completed!")

    print(f"\nView detailed results in MLflow UI:")
    print(f"   1. Run: mlflow ui --backend-store-uri file:./mlruns")
    print(f"   2. Open: http://localhost:5000")
    print(f"   3. Navigate to '{EXPERIMENT_NAME}' experiment")


if __name__ == "__main__":
    main()
