"""
Production MLflow training pipeline for containerized environment.

This pipeline demonstrates clean, production-ready code using the ml_experiments
module for Docker/MLflow setup and model training.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Add the ml_experiments package to the path
sys.path.append(str(Path(__file__).parent.parent))

from ml_experiments import (
    DataConfig,
    DataProcessor,
    ModelConfig,
    ModelEvaluator,
    ModelFactory,
    setup_docker_mlflow_environment,
    setup_mlflow,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_environment() -> str:
    """Setup Docker MLflow environment and create experiment.

    Returns:
        Experiment ID

    Raises:
        RuntimeError: If setup fails
    """
    try:
        setup_docker_mlflow_environment()
        experiment_id = setup_mlflow(
            tracking_uri="http://localhost:5000",
            experiment_name="Real_Estate_Production_Training",
        )
        logger.info(f"MLflow setup complete. Experiment ID: {experiment_id}")
        return experiment_id
    except Exception as e:
        raise RuntimeError(f"Failed to setup MLflow environment: {e}")


def prepare_data() -> tuple:
    """Prepare training and test data.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, metadata)

    Raises:
        RuntimeError: If data preparation fails
    """
    try:
        data_config = DataConfig(
            scaler_type="robust",
            test_size=0.3,
            remove_outliers=True,
            outlier_type="iqr_loose",
            feature_engineering="none",
            random_state=42,
        )

        processor = DataProcessor(
            sales_path=data_config.paths["sales_path"],
            demographics_path=data_config.paths["demographics_path"],
            sales_columns=data_config.sales_columns,
            remove_outliers=data_config.remove_outliers,
            feature_engineering=data_config.feature_engineering,
        )

        X_train, X_test, y_train, y_test = processor.prepare_for_modeling(
            test_size=data_config.test_size, random_state=data_config.random_state
        )

        metadata = processor.get_metadata()
        logger.info(
            f"Data prepared: {metadata['final_samples']} samples, {metadata['final_features']} features"
        )

        return X_train, X_test, y_train, y_test, metadata

    except Exception as e:
        raise RuntimeError(f"Failed to prepare data: {e}")


def train_model(
    model_type: str,
    X_train: Any,
    X_test: Any,
    y_train: Any,
    y_test: Any,
    run_index: int,
) -> Dict[str, Any]:
    """Train and evaluate a single model.

    Args:
        model_type: Type of model to train
        X_train, X_test, y_train, y_test: Training and test data
        run_index: Index for run naming

    Returns:
        Dictionary with training results

    Raises:
        RuntimeError: If training fails
    """
    try:
        # Configure model
        model_config = ModelConfig(
            model_type=model_type, param_set="best", use_feature_selection=False
        )

        # Create model pipeline
        factory = ModelFactory()
        pipeline = factory.create_pipeline(
            model_type=model_config.model_type,
            model_params=model_config.params,
            scaler_type="robust",
            use_feature_selection=model_config.use_feature_selection,
        )

        # Train and evaluate with MLflow tracking
        run_name = f"Production_{model_type.upper()}_{run_index:02d}"

        with mlflow.start_run(run_name=run_name):
            # Train model
            start_time = time.time()
            pipeline.fit(X_train, y_train)
            training_time = time.time() - start_time

            # Evaluate model
            evaluator = ModelEvaluator()
            results = evaluator.evaluate(
                model=pipeline,
                x_train=X_train,
                x_test=X_test,
                y_train=y_train,
                y_test=y_test,
                model_name=f"Production_{model_type.upper()}",
            )

            # Log parameters
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
                    "outlier_handling": "removed",
                    **model_config.params,
                }
            )

            # Log metrics
            train_metrics = results["train_metrics"]
            test_metrics = results["test_metrics"]
            overfitting_metrics = results["overfitting_metrics"]

            for metric_name, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", value)

            for metric_name, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", value)

            for metric_name, value in overfitting_metrics.items():
                mlflow.log_metric(metric_name, value)

            mlflow.log_metric("training_time", training_time)

            # Log model
            input_example = X_train.head(5)
            y_pred_example = pipeline.predict(input_example)
            signature = infer_signature(input_example, y_pred_example)

            model_name = f"production_{model_type}_house_price_model"
            model_info = mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                registered_model_name=model_name,
            )

            # Set tags
            mlflow.set_tags(
                {
                    "model_type": model_type,
                    "experiment_type": "production_training",
                    "framework": "sklearn",
                    "problem_type": "regression",
                    "domain": "real_estate",
                    "version": "1.0",
                    "environment": "docker_mlflow",
                }
            )

            # Prepare results
            results.update(
                {
                    "model_type": model_type,
                    "training_time": training_time,
                    "run_id": mlflow.active_run().info.run_id,
                    "model_uri": model_info.model_uri,
                }
            )

            logger.info(
                f"{model_type.upper()} training complete - Test R²: {test_metrics['r2']:.4f}"
            )

            return results

    except Exception as e:
        raise RuntimeError(f"Failed to train {model_type} model: {e}")


def run_training_pipeline(model_types: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run the complete training pipeline.

    Args:
        model_types: List of model types to train

    Returns:
        Dictionary with all training results

    Raises:
        RuntimeError: If pipeline fails
    """
    if model_types is None:
        model_types = ["lightgbm", "xgboost", "knn"]

    try:
        # Setup environment
        experiment_id = setup_environment()

        # Prepare data
        X_train, X_test, y_train, y_test, metadata = prepare_data()

        # Train models
        all_results = {}
        for i, model_type in enumerate(model_types, 1):
            try:
                results = train_model(
                    model_type=model_type,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    run_index=i,
                )
                all_results[model_type] = results

            except Exception as e:
                logger.error(f"Failed to train {model_type}: {e}")
                continue

        # Summary
        best_model = None
        if all_results:
            logger.info(
                f"Training complete. {len(all_results)} models trained successfully."
            )

            # Log best model
            best_model = max(
                all_results.items(), key=lambda x: x[1]["test_metrics"]["r2"]
            )
            logger.info(
                f"Best model: {best_model[0].upper()} (R²: {best_model[1]['test_metrics']['r2']:.4f})"
            )

        return {
            "experiment_id": experiment_id,
            "metadata": metadata,
            "results": all_results,
            "summary": {
                "models_trained": len(all_results),
                "best_model": best_model[0] if best_model else None,
                "best_r2": best_model[1]["test_metrics"]["r2"] if best_model else None,
            },
        }

    except Exception as e:
        raise RuntimeError(f"Training pipeline failed: {e}")


def main():
    """Main function to run the production training pipeline."""
    try:
        results = run_training_pipeline()

        logger.info("=" * 50)
        logger.info("TRAINING PIPELINE COMPLETE")
        logger.info("=" * 50)
        logger.info(f"Experiment ID: {results['experiment_id']}")
        logger.info(f"Models trained: {results['summary']['models_trained']}")
        logger.info(f"Best model: {results['summary']['best_model']}")
        logger.info(f"Best R²: {results['summary']['best_r2']:.4f}")
        logger.info("MLflow UI: http://localhost:5000")

        return results

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
