"""
Training script that uses the containerized MLflow server.

This script demonstrates how to train models and log them to the dockerized MLflow server
running at http://localhost:5000, eliminating filesystem path issues.
"""

import os
import sys
import time
from typing import Any, Dict, Tuple

import mlflow
import mlflow.data
import mlflow.sklearn

# Add the ml_experiments package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

#######
from dotenv import load_dotenv

from ml_experiments import (
    DataConfig,
    DataProcessor,
    ModelConfig,
    ModelEvaluator,
    ModelFactory,
)

# Load environment variables from .env file
load_dotenv()

# Debug: Check what values are loaded from .env
print("Debug - Values from .env file:")
print(f"  MINIO_ROOT_USER: {os.getenv('MINIO_ROOT_USER')}")
print(f"  MINIO_ROOT_PASSWORD: {os.getenv('MINIO_ROOT_PASSWORD')}")
print(f"  MINIO_PORT: {os.getenv('MINIO_PORT')}")
print(f"  AWS_DEFAULT_REGION: {os.getenv('AWS_DEFAULT_REGION')}")

# Configure MinIO/S3 environment for MLflow artifacts using .env values
minio_user = os.getenv("MINIO_ROOT_USER")
minio_password = os.getenv("MINIO_ROOT_PASSWORD")
minio_port = os.getenv("MINIO_PORT", "9000")
aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
minio_bucket = os.getenv("MINIO_BUCKET")

# Ensure we have the required values
if not minio_user or not minio_password:
    raise ValueError("MINIO_ROOT_USER and MINIO_ROOT_PASSWORD must be set in .env file")

os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://localhost:{minio_port}"
os.environ["AWS_ACCESS_KEY_ID"] = minio_user
os.environ["AWS_SECRET_ACCESS_KEY"] = minio_password
os.environ["AWS_DEFAULT_REGION"] = aws_region
os.environ["BUCKET_NAME"] = minio_bucket or ""
os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

print("Env Configuration loaded:")
print(f"  S3 Endpoint: {os.environ.get('MLFLOW_S3_ENDPOINT_URL')}")
print(f"  Access Key: {os.environ.get('AWS_ACCESS_KEY_ID')}")
print(f"  Secret Key: {os.environ.get('AWS_SECRET_ACCESS_KEY')}")
print(f"  Region: {os.environ.get('AWS_DEFAULT_REGION')}")
print(f"  Ignore TLS: {os.environ.get('MLFLOW_S3_IGNORE_TLS')}")
print(f"  Tracking URI: {os.environ.get('MLFLOW_TRACKING_URI')}")
print(f"  Bucket Name: {os.environ.get('BUCKET_NAME')}")

######


def setup_containerized_mlflow():
    """Setup MLflow to use the containerized server."""

    # Set MLflow tracking URI to the containerized server
    # mlflow_tracking_uri = "http://localhost:5000"
    # mlflow.set_tracking_uri(mlflow_tracking_uri)

    print(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")

    # Test connection
    try:
        experiments = mlflow.search_experiments()
        print(
            f"Successfully connected to MLflow server. Found {len(experiments)} experiments."
        )
        return True
    except Exception as e:
        print(f"Failed to connect to MLflow server: {e}")
        print(
            "Make sure the MLflow container is running with: docker run -d --name mlflow-server -p 5000:5000 ..."
        )
        return False


def create_experiment(experiment_name: str) -> str:
    """Create or get existing experiment."""
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
    except mlflow.exceptions.MlflowException:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")

    mlflow.set_experiment(experiment_name)
    return experiment_id


def prepare_experiment_data() -> Tuple[Any, Any, Any, Any, Dict[str, Any]]:
    """Prepare data for model experiments."""
    print("Preparing data for model training...")

    # Configure data processing
    data_config = DataConfig(
        scaler_type="robust",
        test_size=0.2,
        remove_outliers=True,
        feature_engineering="none",
        random_state=42,
    )

    print(
        f"   Data config: {data_config.scaler_type} scaler, {data_config.test_size} test split"
    )
    print(f"   Outlier removal: {data_config.remove_outliers}")
    print(f"   Feature engineering: {data_config.feature_engineering}")

    # Load and prepare data
    data_processor = DataProcessor(
        sales_path=data_config.paths["sales_path"],
        demographics_path=data_config.paths["demographics_path"],
        sales_columns=data_config.sales_columns,
        remove_outliers=data_config.remove_outliers,
        feature_engineering=data_config.feature_engineering,
    )

    X_train, X_test, y_train, y_test = data_processor.prepare_for_modeling(
        test_size=data_config.test_size, random_state=data_config.random_state
    )

    metadata = data_processor.get_metadata()
    print(
        f"   Data loaded: {metadata['final_samples']} samples, {metadata['final_features']} features"
    )
    print(
        f"   Train/test split: {metadata['train_samples']}/{metadata['test_samples']}"
    )

    return X_train, X_test, y_train, y_test, metadata


def train_and_log_model(
    model_type: str,
    X_train: Any,
    X_test: Any,
    y_train: Any,
    y_test: Any,
    run_index: int,
) -> Dict[str, Any]:
    """Train a model and log it to the containerized MLflow server."""
    print(f"\n--- Training {model_type.upper()} Model ---")

    # Configure model
    model_config = ModelConfig(
        model_type=model_type, param_set="best", use_feature_selection=False
    )

    print(f"   Model parameters: {len(model_config.params)} configured")

    # Create model pipeline
    model_factory = ModelFactory()
    pipeline = model_factory.create_pipeline(
        model_type=model_config.model_type,
        model_params=model_config.params,
        scaler_type="robust",
        use_feature_selection=model_config.use_feature_selection,
    )

    print(f"   Pipeline created with {len(pipeline.steps)} steps")

    # Train and evaluate with MLflow tracking
    run_name = f"Docker_MLflow_{model_type.upper()}_{run_index:02d}"

    with mlflow.start_run(run_name=run_name):
        print(f"   Started MLflow run: {run_name}")

        # Record training time
        start_time = time.time()

        # Train model
        pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time
        print(f"   Training completed in {training_time:.2f} seconds")

        # Evaluate model
        evaluator = ModelEvaluator()
        results = evaluator.evaluate(
            model=pipeline,
            x_train=X_train,
            x_test=X_test,
            y_train=y_train,
            y_test=y_test,
            model_name=f"Docker_{model_type.upper()}",
        )

        # Add training time to results
        results["training_time"] = training_time
        results["model_type"] = model_type

        # Display key metrics
        train_metrics = results["train_metrics"]
        test_metrics = results["test_metrics"]
        overfitting_metrics = results["overfitting_metrics"]

        print(f"Train R²: {train_metrics['r2']:.4f}")
        print(f"Test R²: {test_metrics['r2']:.4f}")
        print(f"Test MAE: ${test_metrics['mae']:,.2f}")
        print(
            f"   Overfitting ratio: {overfitting_metrics['overfitting_ratio_r2']:.4f}"
        )

        # Log parameters to MLflow
        print("   Logging parameters to MLflow...")
        mlflow.log_params(
            {
                "model_type": model_type,
                "test_size": 0.2,
                "scaler_type": "robust",
                "dataset_size": len(X_train) + len(X_test),
                "n_features": len(X_train.columns),
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "feature_engineering": "none",
                "outlier_handling": "removed" if True else "kept",
                "mlflow_server": "containerized",
                **model_config.params,
            }
        )

        # Log metrics to MLflow
        print("   Logging metrics to MLflow...")
        for metric_name, value in train_metrics.items():
            mlflow.log_metric(f"train_{metric_name}", value)

        for metric_name, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)

        for metric_name, value in overfitting_metrics.items():
            mlflow.log_metric(metric_name, value)

        mlflow.log_metric("training_time", training_time)

        # Log datasets to MLflow
        print("   Logging datasets to MLflow...")

        # Create and log training dataset
        train_dataset = mlflow.data.from_pandas(
            X_train.join(y_train),
            source="kc_house_data_processed",
            name="training_data",
            targets="price",
        )
        mlflow.log_input(train_dataset, context="training")

        # Create and log test dataset
        test_dataset = mlflow.data.from_pandas(
            X_test.join(y_test),
            source="kc_house_data_processed",
            name="test_data",
            targets="price",
        )
        mlflow.log_input(test_dataset, context="validation")

        # Log the model with proper signature and metadata
        print("   Logging model to containerized MLflow...")

        import json

        from mlflow.models import infer_signature

        # Create model signature
        input_example = X_train.head(5)
        y_pred_example = pipeline.predict(input_example)
        signature = infer_signature(input_example, y_pred_example)
        print(f"   Model signature created: {signature}")

        # Log model to MLflow (this will be stored in the container)
        model_name = f"docker_{model_type}_house_price_model"

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name=model_name,
            signature=signature,
            input_example=input_example,
            registered_model_name=model_name,
        )

        print("MLflow model logged successfully!")
        print(f"Model URI: {model_info.model_uri}")
        print(f"Model artifact path: {model_info.artifact_path}")
        print(f"Model UUID: {model_info.model_uuid}")

        # Set comprehensive tags
        mlflow.set_tags(
            {
                "model_type": model_type,
                "experiment_type": "docker_mlflow_demo",
                "framework": "sklearn",
                "problem_type": "regression",
                "domain": "real_estate",
                "data_source": "kc_house_data",
                "preprocessing": "robust_scaler",
                "version": "1.0",
                "author": "docker_mlflow_demo",
                "purpose": "containerized_mlflow_test",
                "mlflow_server": "containerized",
                "storage_type": "container_internal",
                "model_uri": model_info.model_uri,
                "artifact_path": model_info.artifact_path,
            }
        )

        # Create and log model metadata
        run_id = mlflow.active_run().info.run_id
        model_metadata = {
            "model_info": {
                "model_type": model_type,
                "model_uri": model_info.model_uri,
                "model_name": model_name,
                "artifact_path": model_info.artifact_path,
                "run_id": run_id,
                "model_uuid": model_info.model_uuid,
            },
            "mlflow_info": {
                "tracking_uri": mlflow.get_tracking_uri(),
                "experiment_id": mlflow.active_run().info.experiment_id,
                "run_id": run_id,
                "storage_type": "container_internal",
            },
            "data_info": {
                "features": list(X_train.columns),
                "feature_count": len(X_train.columns),
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "data_source": "kc_house_data",
                "preprocessing": "robust_scaler",
            },
            "performance": {
                "train_metrics": {k: float(v) for k, v in train_metrics.items()},
                "test_metrics": {k: float(v) for k, v in test_metrics.items()},
                "overfitting_metrics": {
                    k: float(v) for k, v in overfitting_metrics.items()
                },
                "training_time_seconds": training_time,
            },
            "model_config": {
                "parameters": model_config.params,
                "scaler_type": "robust",
                "feature_selection": model_config.use_feature_selection,
            },
        }

        # Save and log metadata
        metadata_file = f"docker_model_metadata_{model_type}.json"
        with open(metadata_file, "w") as f:
            json.dump(model_metadata, f, indent=2)
        mlflow.log_artifact(metadata_file)

        # Clean up local file
        os.remove(metadata_file)

        print("   Model and metadata logged to containerized MLflow server!")

        return results


def test_model_loading(model_name: str, run_id: str):
    """Test loading a model from the containerized MLflow server."""
    print("\n--- Testing Model Loading ---")
    print(f"   Attempting to load model from run: {run_id}")

    try:
        # Load model using run ID (this should work with containerized MLflow)
        model_uri = f"runs:/{run_id}/model"
        loaded_model = mlflow.sklearn.load_model(model_uri)

        print(f"   SUCCESS: Model loaded from: {model_uri}")
        print(f"   Model type: {type(loaded_model)}")

        # Test prediction with dummy data
        import pandas as pd

        # Create dummy test data (matching expected features)
        dummy_data = pd.DataFrame(
            {
                "bedrooms": [3],
                "bathrooms": [2.0],
                "sqft_living": [2000],
                "sqft_lot": [5000],
                "floors": [1.0],
                "waterfront": [0],
                "view": [0],
                "condition": [3],
                "grade": [7],
                "sqft_above": [2000],
                "sqft_basement": [0],
                "yr_built": [1990],
                "yr_renovated": [0],
                "zipcode": [98028],
                "lat": [47.7749],
                "long": [-122.2444],
                "sqft_living15": [1800],
                "sqft_lot15": [4500],
            }
        )

        # Make prediction
        try:
            prediction = loaded_model.predict(dummy_data)
            print(f"   SUCCESS: Test prediction: ${prediction[0]:,.2f}")
            return True
        except Exception as e:
            print(f"   FAILED: Prediction error: {e}")
            return False

    except Exception as e:
        print(f"   FAILED: Model loading error: {e}")
        return False


def main():
    """Main function to run the containerized MLflow training demo."""
    print("=" * 80)
    print("CONTAINERIZED MLFLOW TRAINING DEMO")
    print("=" * 80)
    print("This demo trains models using a containerized MLflow server")
    print("eliminating filesystem path issues.")
    print("=" * 80)

    # Setup containerized MLflow
    if not setup_containerized_mlflow():
        print("ERROR: Cannot connect to MLflow server. Exiting.")
        return

    # Create experiment
    experiment_name = "MLflow_Real_Estate_Demo_2"
    _ = create_experiment(experiment_name)

    try:
        # Prepare data
        X_train, X_test, y_train, y_test, metadata = prepare_experiment_data()

        # Train models
        models_to_test = ["lightgbm", "xgboost"]  # Start with 2 models for demo
        all_results = []
        trained_runs = []

        print(f"\nTraining {len(models_to_test)} models with containerized MLflow...")

        for i, model_type in enumerate(models_to_test, 1):
            try:
                results = train_and_log_model(
                    model_type=model_type,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    run_index=i,
                )
                all_results.append(results)

                # Store run info for testing model loading
                run_id = (
                    mlflow.active_run().info.run_id if mlflow.active_run() else None
                )
                if run_id:
                    trained_runs.append((model_type, run_id))

            except Exception as e:
                print(f"   ERROR: Failed to train {model_type}: {e}")
                continue

        # Display results
        if all_results:
            print("\n" + "=" * 80)
            print("TRAINING RESULTS")
            print("=" * 80)

            print(
                f"\n{'Model':<12} {'Train R²':<10} {'Test R²':<10} {'Test MAE':<12} {'Time (s)':<10}"
            )
            print("-" * 60)

            for result in all_results:
                model_type = result["model_type"]
                train_r2 = result["train_metrics"]["r2"]
                test_r2 = result["test_metrics"]["r2"]
                test_mae = result["test_metrics"]["mae"]
                training_time = result["training_time"]

                print(
                    f"{model_type.upper():<12} {train_r2:<10.4f} {test_r2:<10.4f} ${test_mae:<11,.0f} {training_time:<10.2f}"
                )

        # Test model loading
        if trained_runs:
            print("\n" + "=" * 80)
            print("TESTING MODEL LOADING FROM CONTAINERIZED MLFLOW")
            print("=" * 80)

            for model_type, run_id in trained_runs:
                success = test_model_loading(model_type, run_id)
                if success:
                    print(f"   SUCCESS: {model_type.upper()} model loading")
                else:
                    print(f"   FAILED: {model_type.upper()} model loading")

        print("\n" + "=" * 80)
        print("DEMO SUMMARY")
        print("=" * 80)
        print(f"MLflow server: {mlflow.get_tracking_uri()}")
        print(f"Experiment: {experiment_name}")
        print(f"Models trained: {len(all_results)}")
        print(f"Data samples: {metadata['final_samples']}")
        print(f"Features: {metadata['final_features']}")
        print("All models stored in containerized MLflow server")
        print("No filesystem path issues!")
        print("\nContainerized MLflow training completed successfully!")
        print("\nAccess MLflow UI at: http://localhost:5000")

    except FileNotFoundError as e:
        print(f"ERROR: Data files not found: {e}")
        print("This demo requires the KC House dataset.")
        print("In a real scenario, ensure data files are available.")

    except Exception as e:
        print(f"ERROR: Training failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
