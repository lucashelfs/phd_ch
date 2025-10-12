import json
import pathlib
import pickle
from typing import List, Tuple

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
from sklearn import model_selection
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature
import lightgbm as lgb

# Configuration
SALES_PATH = "data/kc_house_data.csv"  # path to CSV with home sale data
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"  # path to CSV with demographics
# List of columns (subset) that will be taken from home sale data
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
OUTPUT_DIR = "model"  # Directory where output artifacts will be saved

# MLflow Configuration
EXPERIMENT_NAME = "Real Estate Price Prediction"
TRACKING_URI = (
    "file:./mlruns"  # Local file-based tracking (change to server URI if needed)
)


def load_data(
    sales_path: str, demographics_path: str, sales_column_selection: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the target and feature data by merging sales and demographics.

    Args:
        sales_path: path to CSV file with home sale data
        demographics_path: path to CSV file with demographics data
        sales_column_selection: list of columns from sales data to be used as
            features

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


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Calculate regression metrics.

    Args:
        y_true: True target values
        y_pred: Predicted values

    Returns:
        Dictionary containing calculated metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


def log_data_info(x: pd.DataFrame, y: pd.Series) -> None:
    """Log dataset information to MLflow.

    Args:
        x: Feature DataFrame
        y: Target Series
    """
    # Log dataset statistics
    mlflow.log_params(
        {
            "dataset_size": len(x),
            "n_features": len(x.columns),
            "target_mean": float(y.mean()),
            "target_std": float(y.std()),
            "target_min": float(y.min()),
            "target_max": float(y.max()),
        }
    )

    # Log feature names
    feature_info = {
        "features": list(x.columns),
        "feature_count": len(x.columns),
        "numeric_features": list(x.select_dtypes(include=[np.number]).columns),
        "categorical_features": list(x.select_dtypes(exclude=[np.number]).columns),
    }

    # Save feature info as artifact
    with open("feature_info.json", "w") as f:
        json.dump(feature_info, f, indent=2)
    mlflow.log_artifact("feature_info.json")


def main():
    """Load data, train LightGBM model with MLflow tracking, and export artifacts."""

    # Set MLflow tracking URI and experiment
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Enable autologging for LightGBM
    mlflow.lightgbm.autolog(log_input_examples=True, log_model_signatures=True)

    print(f"Loading data from {SALES_PATH} and {DEMOGRAPHICS_PATH}")
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)

    print(f"Dataset loaded: {len(x)} samples, {len(x.columns)} features")
    print(f"Target statistics: mean=${y.mean():.2f}, std=${y.std():.2f}")

    # Split data for proper evaluation
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # LightGBM model parameters
    model_params = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "objective": "regression",
        "metric": "rmse",
        "verbose": -1,  # Suppress LightGBM warnings
    }

    scaler_params = {
        "quantile_range": (25.0, 75.0),
        "with_centering": True,
        "with_scaling": True,
    }

    # Start MLflow run
    with mlflow.start_run(run_name="LightGBM_Real_Estate_Prediction") as run:
        print(f"MLflow run started: {run.info.run_id}")

        # Log dataset information
        log_data_info(x, y)

        # Log model parameters
        mlflow.log_params(
            {
                **{f"lgb_{k}": v for k, v in model_params.items()},
                **{f"scaler_{k}": v for k, v in scaler_params.items()},
                "test_size": 0.2,
                "random_state": 42,
            }
        )

        # Create and train model pipeline
        print("Training LightGBM model...")
        model = pipeline.make_pipeline(
            preprocessing.RobustScaler(**scaler_params),
            lgb.LGBMRegressor(**model_params),
        )

        model.fit(x_train, y_train)

        # Make predictions
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        # Calculate metrics
        train_metrics = calculate_metrics(y_train, y_train_pred)
        test_metrics = calculate_metrics(y_test, y_test_pred)

        print(
            f"Training Metrics - MAE: ${train_metrics['mae']:.2f}, "
            f"RMSE: ${train_metrics['rmse']:.2f}, R²: {train_metrics['r2']:.4f}"
        )
        print(
            f"Test Metrics - MAE: ${test_metrics['mae']:.2f}, "
            f"RMSE: ${test_metrics['rmse']:.2f}, R²: {test_metrics['r2']:.4f}"
        )

        # Log metrics to MLflow
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

        # Create model signature
        signature = infer_signature(x_train, y_train_pred)

        # Log the model with MLflow (autolog will handle this, but we can add custom info)
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=x_train.head(5),
            registered_model_name="real_estate_lightgbm_model",
        )

        # Set tags for better organization
        mlflow.set_tags(
            {
                "model_type": "LGBMRegressor",
                "preprocessing": "RobustScaler",
                "problem_type": "regression",
                "domain": "real_estate",
                "data_source": "kc_house_data",
                "author": "MLflow Enhanced Training",
                "version": "1.0",
            }
        )

        # Save traditional artifacts for backward compatibility
        output_dir = pathlib.Path(OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)

        # Save model pickle
        pickle.dump(model, open(output_dir / "model.pkl", "wb"))

        # Save feature list
        json.dump(list(x_train.columns), open(output_dir / "model_features.json", "w"))

        # Log traditional artifacts
        mlflow.log_artifacts(str(output_dir), artifact_path="traditional_artifacts")

        # Create and log evaluation report
        evaluation_report = {
            "model_type": "LGBMRegressor with RobustScaler",
            "training_samples": len(x_train),
            "test_samples": len(x_test),
            "features": list(x_train.columns),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "model_params": model_params,
            "scaler_params": scaler_params,
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
        }

        with open("evaluation_report.json", "w") as f:
            json.dump(evaluation_report, f, indent=2)
        mlflow.log_artifact("evaluation_report.json")

        print(f"\nMLflow tracking completed!")
        print(f"Run ID: {run.info.run_id}")
        print(f"Experiment ID: {run.info.experiment_id}")
        print(f"Model URI: {model_info.model_uri}")
        print(f"Registered Model: real_estate_lightgbm_model")

        # Print MLflow UI access information
        print(f"\nView results in MLflow UI:")
        print(f"URL: {TRACKING_URI}")
        print(f"Experiment: {EXPERIMENT_NAME}")


if __name__ == "__main__":
    main()
