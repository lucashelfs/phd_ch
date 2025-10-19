"""
Baseline Model Pipeline with MLflow Integration

This script is based on the original create_model.py but adds MLflow logging
to track experiments and model performance. It uses the same simple approach
with KNeighborsRegressor and RobustScaler.
"""

import json
import os
import pathlib
import pickle
from typing import List, Tuple

import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
import pandas
from dotenv import load_dotenv
from sklearn import model_selection, neighbors, pipeline, preprocessing
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)


def weighted_mean_absolute_percentage_error(y_true, y_pred):
    """Calculate Weighted Mean Absolute Percentage Error (WMAPE).

    WMAPE is more robust than MAPE as it weights errors by the magnitude of actual values.

    Args:
        y_true: True target values
        y_pred: Predicted values

    Returns:
        WMAPE as a percentage (0-100)
    """
    return (sum(abs(y_true - y_pred)) / sum(abs(y_true))) * 100


# Load environment variables
load_dotenv()

# Data paths (same as original)
SALES_PATH = "data/kc_house_data.csv"
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"
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

# MLflow configuration
EXPERIMENT_NAME = "Baseline Model"


def setup_mlflow():
    """Setup MLflow connection to the containerized server."""
    # Configure MLflow tracking URI
    mlflow_tracking_uri = "http://localhost:5000"
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Configure MinIO/S3 for artifact storage
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = (
        f"http://localhost:{os.getenv('MINIO_PORT', '9000')}"
    )
    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("MINIO_ROOT_USER", "minio")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("MINIO_ROOT_PASSWORD", "minio123")
    os.environ["AWS_DEFAULT_REGION"] = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"

    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    # Create or get experiment
    try:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
        print(f"Created new experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")
    except MlflowException:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment:
            experiment_id = experiment.experiment_id
            print(f"Using existing experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")
        else:
            raise ValueError(f"Could not create or find experiment: {EXPERIMENT_NAME}")

    mlflow.set_experiment(EXPERIMENT_NAME)
    return experiment_id


def load_data(
    sales_path: str, demographics_path: str, sales_column_selection: List[str]
) -> Tuple[pandas.DataFrame, pandas.Series]:
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
    data = pandas.read_csv(
        sales_path, usecols=sales_column_selection, dtype={"zipcode": str}
    )
    demographics = pandas.read_csv(demographics_path, dtype={"zipcode": str})

    merged_data = data.merge(demographics, how="left", on="zipcode").drop(
        columns="zipcode"
    )
    # Remove the target variable from the dataframe, features will remain
    y = merged_data.pop("price")
    x = merged_data

    return x, y


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate model performance and return metrics."""
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_metrics = {
        "r2": r2_score(y_train, y_train_pred),
        "mae": mean_absolute_error(y_train, y_train_pred),
        "rmse": mean_squared_error(y_train, y_train_pred, squared=False),
        "mape": mean_absolute_percentage_error(y_train, y_train_pred)
        * 100,  # Convert to percentage
        "wmape": weighted_mean_absolute_percentage_error(y_train, y_train_pred),
    }

    test_metrics = {
        "r2": r2_score(y_test, y_test_pred),
        "mae": mean_absolute_error(y_test, y_test_pred),
        "rmse": mean_squared_error(y_test, y_test_pred, squared=False),
        "mape": mean_absolute_percentage_error(y_test, y_test_pred)
        * 100,  # Convert to percentage
        "wmape": weighted_mean_absolute_percentage_error(y_test, y_test_pred),
    }

    return train_metrics, test_metrics


def main():
    """Load data, train model, log to MLflow, and export artifacts."""
    print("Starting Baseline Model Pipeline with MLflow Integration")
    print("=" * 60)

    # Setup MLflow
    _ = setup_mlflow()

    # Load data (same as original)
    print("Loading and preparing data...")
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    print(f"Data loaded: {len(x)} total samples")
    print(f"Features: {len(x.columns)}")
    print(f"Train/Test split: {len(x_train)}/{len(x_test)}")

    # Start MLflow run
    with mlflow.start_run(run_name="Baseline_KNN_Model"):
        print("\nTraining model...")

        # Create and train model (same as original)
        model = pipeline.make_pipeline(
            preprocessing.RobustScaler(), neighbors.KNeighborsRegressor()
        ).fit(x_train, y_train)

        print("Model training completed!")

        # Evaluate model
        print("Evaluating model performance...")
        train_metrics, test_metrics = evaluate_model(
            model, x_train, x_test, y_train, y_test
        )

        # Display metrics
        print("\nModel Performance:")
        print(f"Train R²: {train_metrics['r2']:.4f}")
        print(f"Test R²:  {test_metrics['r2']:.4f}")
        print(f"Train MAE: ${train_metrics['mae']:,.2f}")
        print(f"Test MAE:  ${test_metrics['mae']:,.2f}")
        print(f"Train RMSE: ${train_metrics['rmse']:,.2f}")
        print(f"Test RMSE:  ${test_metrics['rmse']:,.2f}")
        print(f"Train MAPE: {train_metrics['mape']:.2f}%")
        print(f"Test MAPE:  {test_metrics['mape']:.2f}%")
        print(f"Train WMAPE: {train_metrics['wmape']:.2f}%")
        print(f"Test WMAPE:  {test_metrics['wmape']:.2f}%")

        # Log parameters to MLflow
        mlflow.log_params(
            {
                "model_type": "KNeighborsRegressor",
                "scaler_type": "RobustScaler",
                "test_size": 0.2,
                "random_state": 42,
                "n_neighbors": 5,  # default value
                "dataset_size": len(x),
                "n_features": len(x.columns),
                "train_samples": len(x_train),
                "test_samples": len(x_test),
            }
        )

        # Log metrics to MLflow
        for metric_name, value in train_metrics.items():
            mlflow.log_metric(f"train_{metric_name}", value)

        for metric_name, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)

        # Log overfitting metrics
        overfitting_r2 = train_metrics["r2"] - test_metrics["r2"]
        overfitting_ratio_r2 = (
            train_metrics["r2"] / test_metrics["r2"] if test_metrics["r2"] != 0 else 0
        )

        # MAPE overfitting metrics
        overfitting_mape = test_metrics["mape"] - train_metrics["mape"]
        overfitting_ratio_mape = (
            test_metrics["mape"] / train_metrics["mape"]
            if train_metrics["mape"] != 0
            else 0
        )

        # WMAPE overfitting metrics
        overfitting_wmape = test_metrics["wmape"] - train_metrics["wmape"]
        overfitting_ratio_wmape = (
            test_metrics["wmape"] / train_metrics["wmape"]
            if train_metrics["wmape"] != 0
            else 0
        )

        # Log overfitting metrics with both naming conventions for consistency
        mlflow.log_metric("overfitting_r2_diff", overfitting_r2)
        mlflow.log_metric("overfitting_ratio_r2", overfitting_ratio_r2)
        mlflow.log_metric("overfitting_mape_diff", overfitting_mape)
        mlflow.log_metric("overfitting_ratio_mape", overfitting_ratio_mape)
        mlflow.log_metric("overfitting_wmape_diff", overfitting_wmape)
        mlflow.log_metric("overfitting_ratio_wmape", overfitting_ratio_wmape)

        # Add gap metrics to match ml_experiments naming convention
        mae_gap = test_metrics["mae"] - train_metrics["mae"]
        mape_gap = test_metrics["mape"] - train_metrics["mape"]
        wmape_gap = test_metrics["wmape"] - train_metrics["wmape"]

        mlflow.log_metric("mae_gap", mae_gap)
        mlflow.log_metric("mape_gap", mape_gap)
        mlflow.log_metric("wmape_gap", wmape_gap)

        # Display overfitting metrics
        print(f"Overfitting R² diff: {overfitting_r2:.4f}")
        print(f"Overfitting R² ratio: {overfitting_ratio_r2:.4f}")
        print(f"Overfitting MAPE diff: {overfitting_mape:.2f}%")
        print(f"Overfitting MAPE ratio: {overfitting_ratio_mape:.4f}")
        print(f"Overfitting WMAPE diff: {overfitting_wmape:.2f}%")
        print(f"Overfitting WMAPE ratio: {overfitting_ratio_wmape:.4f}")

        # Create model signature for MLflow
        input_example = x_train.head(5)
        predictions = model.predict(input_example)
        signature = infer_signature(input_example, predictions)

        # Log model to MLflow
        print("\nLogging model to MLflow...")
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name="baseline_knn_house_price_model",
        )

        # Set tags
        mlflow.set_tags(
            {
                "model_type": "KNeighborsRegressor",
                "experiment_type": "baseline",
                "framework": "sklearn",
                "problem_type": "regression",
                "domain": "real_estate",
                "data_source": "kc_house_data",
                "preprocessing": "robust_scaler",
                "version": "1.0",
                "purpose": "baseline_model",
            }
        )

        # Save traditional artifacts (same as original)
        print("Saving traditional model artifacts...")
        output_dir = pathlib.Path(OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)

        # Output model artifacts: pickled model and JSON list of features
        pickle.dump(model, open(output_dir / "model.pkl", "wb"))
        json.dump(list(x_train.columns), open(output_dir / "model_features.json", "w"))

        # Log traditional artifacts to MLflow as well
        mlflow.log_artifact(str(output_dir / "model.pkl"))
        mlflow.log_artifact(str(output_dir / "model_features.json"))

        current_run = mlflow.active_run()
        if current_run:
            print(f"\nMLflow Model URI: {model_info.model_uri}")
            print(f"Run ID: {current_run.info.run_id}")

    print("\n" + "=" * 60)
    print("Baseline Model Pipeline Completed Successfully!")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print("Check MLflow UI at: http://localhost:5000")
    print("=" * 60)


if __name__ == "__main__":
    main()
