"""
Example usage of the modularized ML experiments package.

This script demonstrates how to use the new modular structure to run
comprehensive model comparison experiments across KNN, XGBoost, and LightGBM.
"""

import os
import sys
import time
from typing import Any, Dict, List, Tuple

import mlflow
import mlflow.data
import mlflow.sklearn

# Add the ml_experiments package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

from ml_experiments import (
    DataConfig,
    DataProcessor,
    MLflowConfig,
    ModelConfig,
    ModelEvaluator,
    ModelFactory,
)
from ml_experiments.core.mlflow_utils import (
    setup_mlflow,
)


def prepare_experiment_data() -> Tuple[Any, Any, Any, Any, Dict[str, Any]]:
    """Prepare data for all model experiments.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, metadata)
    """
    print("Preparing data for model comparison...")

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
    try:
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

    except FileNotFoundError as e:
        print(f"   Data files not found: {e}")
        print("   This is expected in the demo - data files are not included")
        print("   In a real scenario, you would have the data files available")
        raise


def test_single_model(
    model_type: str,
    X_train: Any,
    X_test: Any,
    y_train: Any,
    y_test: Any,
    mlflow_config: MLflowConfig,
    evaluator: ModelEvaluator,
    run_index: int,
) -> Dict[str, Any]:
    """Test a single model and return results.

    Args:
        model_type: Type of model to test
        X_train, X_test, y_train, y_test: Data splits
        mlflow_config: MLflow configuration
        evaluator: Model evaluator instance
        run_index: Index for run naming

    Returns:
        Dictionary with model results
    """
    print(f"\n--- Testing {model_type.upper()} Model ---")

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
        scaler_type="robust",  # Consistent scaler across all models
        use_feature_selection=model_config.use_feature_selection,
    )

    print(f"   Pipeline created with {len(pipeline.steps)} steps")

    # Setup MLflow experiment
    setup_mlflow(
        tracking_uri=mlflow_config.tracking_config["tracking_uri"],
        experiment_name=mlflow_config.experiment_name,
    )

    # Train and evaluate with direct MLflow tracking
    with mlflow.start_run(
        run_name=mlflow_config.create_run_name(model_type, run_index)
    ):
        # Record training time
        start_time = time.time()

        # Train model
        pipeline.fit(X_train, y_train)

        training_time = time.time() - start_time
        print(f"   Training completed in {training_time:.2f} seconds")

        # Evaluate model
        results = evaluator.evaluate(
            model=pipeline,
            x_train=X_train,
            x_test=X_test,
            y_train=y_train,
            y_test=y_test,
            model_name=f"{model_type.upper()}_Comparison",
        )

        # Add training time to results
        results["training_time"] = training_time
        results["model_type"] = model_type

        # Display key metrics
        train_metrics = results["train_metrics"]
        test_metrics = results["test_metrics"]
        overfitting_metrics = results["overfitting_metrics"]

        print(f"   Train R²: {train_metrics['r2']:.4f}")
        print(f"   Test R²: {test_metrics['r2']:.4f}")
        print(f"   Test MAE: ${test_metrics['mae']:,.2f}")
        print(
            f"   Overfitting ratio: {overfitting_metrics['overfitting_ratio_r2']:.4f}"
        )

        # Log parameters directly to MLflow
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
                "outlier_handling": "keep",
                **model_config.params,
            }
        )

        # Log metrics directly to MLflow
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

        print("   Datasets logged successfully")

        # Log comprehensive model with all metadata
        print("   Logging complete MLflow model with metadata...")

        import json
        import pickle

        from mlflow.models import infer_signature

        # 1. Create proper model signature from training data
        input_example = X_train.head(5)
        y_pred_example = pipeline.predict(input_example)
        signature = infer_signature(input_example, y_pred_example)
        print(f"   Model signature created: {signature}")

        # 2. Log the complete model with all metadata (PRIMARY MODEL LOGGING)
        model_name = f"{model_type}_house_price_model"

        # Log model directly to MLflow (this will appear in UI)
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=model_name,
        )

        print(f"   MLflow model logged: {model_info.model_uri}")
        print(f"   Model registered as: {model_name}")
        print(f"   Model UUID: {model_info.model_uuid}")

        # 3. Set comprehensive experiment tags
        mlflow.set_tags(
            {
                "model_type": model_type,
                "experiment_type": "model_comparison",
                "framework": "sklearn",
                "problem_type": "regression",
                "domain": "real_estate",
                "data_source": "kc_house_data",
                "preprocessing": "robust_scaler",
                "version": "1.0",
                "author": "ml_experiments_framework",
                "purpose": "house_price_prediction",
                "model_uri": model_info.model_uri,
                "registered_model": model_name,
            }
        )

        # 4. Create and log comprehensive model metadata
        run_id = mlflow.active_run().info.run_id
        model_metadata = {
            "model_info": {
                "model_type": model_type,
                "model_uri": model_info.model_uri,
                "registered_name": model_name,
                "artifact_path": "model",
                "run_id": run_id,
            },
            "data_info": {
                "features": list(X_train.columns),
                "feature_count": len(X_train.columns),
                "numeric_features": list(
                    X_train.select_dtypes(include=["number"]).columns
                ),
                "categorical_features": list(
                    X_train.select_dtypes(exclude=["number"]).columns
                ),
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
            "signature": {
                "inputs": str(signature.inputs) if signature else None,
                "outputs": str(signature.outputs) if signature else None,
            },
        }

        with open("model_metadata.json", "w") as f:
            json.dump(model_metadata, f, indent=2)
        mlflow.log_artifact("model_metadata.json")

        return results


def display_comparison_results(all_results: List[Dict[str, Any]]) -> None:
    """Display comprehensive comparison of all model results.

    Args:
        all_results: List of results from all models
    """
    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)

    # Create comparison table
    print(
        f"\n{'Model':<12} {'Train R²':<10} {'Test R²':<10} {'Test MAE':<12} {'Overfitting':<12} {'Time (s)':<10}"
    )
    print("-" * 80)

    best_test_r2 = 0
    best_model = None

    for result in all_results:
        model_type = result["model_type"]
        train_r2 = result["train_metrics"]["r2"]
        test_r2 = result["test_metrics"]["r2"]
        test_mae = result["test_metrics"]["mae"]
        overfitting_ratio = result["overfitting_metrics"]["overfitting_ratio_r2"]
        training_time = result["training_time"]

        # Track best model
        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_model = result

        print(
            f"{model_type.upper():<12} {train_r2:<10.4f} {test_r2:<10.4f} ${test_mae:<11,.0f} {overfitting_ratio:<12.4f} {training_time:<10.2f}"
        )

    # Display best model
    print("\n" + "=" * 80)
    print("BEST MODEL ANALYSIS")
    print("=" * 80)

    if best_model:
        model_type = best_model["model_type"]
        summary = best_model["summary"]

        print(f"\nBest performing model: {model_type.upper()}")
        print(f"Performance level: {summary['performance_level']}")
        print(f"Overfitting level: {summary['overfitting_level']}")
        print(f"Test R²: {summary['test_r2']:.4f}")
        print(f"Test MAE: ${summary['test_mae']:,.2f}")
        print(f"Training time: {best_model['training_time']:.2f} seconds")

        # Performance interpretation
        print("\nPerformance Interpretation:")
        if summary["test_r2"] >= 0.8:
            print("   Excellent model performance - explains >80% of variance")
        elif summary["test_r2"] >= 0.7:
            print("   Good model performance - explains >70% of variance")
        elif summary["test_r2"] >= 0.6:
            print("   Fair model performance - explains >60% of variance")
        else:
            print("   Model performance could be improved")

        if summary["overfitting_level"] == "Minimal":
            print("   Low overfitting risk - model generalizes well")
        elif summary["overfitting_level"] == "Moderate":
            print("   Moderate overfitting - monitor on new data")
        else:
            print("   High overfitting risk - consider regularization")


def run_comprehensive_model_comparison():
    """Run comprehensive comparison of all three models."""

    print("ML EXPERIMENTS - COMPREHENSIVE MODEL COMPARISON")
    print("=" * 80)
    print("Testing KNN, XGBoost, and LightGBM models")
    print("=" * 80)

    # Configure MLflow tracking
    mlflow_config = MLflowConfig(
        tracking_config="local",
        experiment_type="model_comparison",
        autolog_config="minimal",
    )
    print(f"MLflow experiment: {mlflow_config.experiment_name}")

    try:
        # Prepare data once for all models
        X_train, X_test, y_train, y_test, metadata = prepare_experiment_data()

        # Initialize evaluator
        evaluator = ModelEvaluator()

        # Test all models
        models_to_test = ["knn", "xgboost", "lightgbm"]
        all_results = []

        print(f"\nTesting {len(models_to_test)} models...")

        for i, model_type in enumerate(models_to_test, 1):
            try:
                results = test_single_model(
                    model_type=model_type,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    mlflow_config=mlflow_config,
                    evaluator=evaluator,
                    run_index=i,
                )
                all_results.append(results)

            except Exception as e:
                print(f"   Error testing {model_type}: {e}")
                continue

        # Display comprehensive comparison
        if all_results:
            display_comparison_results(all_results)

            print("\n" + "=" * 80)
            print("EXPERIMENT SUMMARY")
            print("=" * 80)
            print(f"Total models tested: {len(all_results)}")
            print(f"Data samples: {metadata['final_samples']}")
            print(f"Features: {metadata['final_features']}")
            print(f"MLflow experiment: {mlflow_config.experiment_name}")
            print("All results logged to MLflow for detailed analysis")
        else:
            print("No models were successfully tested")

    except FileNotFoundError:
        print("\nData files not available - running configuration demo instead...")
        demonstrate_configurations()

    except Exception as e:
        print(f"Experiment failed: {e}")
        print("Running configuration demo instead...")
        demonstrate_configurations()


def demonstrate_configurations():
    """Demonstrate different configuration options."""

    print("\n" + "=" * 60)
    print("Demonstrating Configuration Options")
    print("=" * 60)

    # Different data configurations
    print("\nData Configuration Options:")

    configs = [
        DataConfig(scaler_type="robust", test_size=0.2, remove_outliers=False),
        DataConfig(scaler_type="standard", test_size=0.25, remove_outliers=True),
        DataConfig(scaler_type="minmax", test_size=0.3, feature_engineering="poly2"),
    ]

    for i, config in enumerate(configs, 1):
        print(
            f"   Config {i}: {config.scaler_type} scaler, {config.test_size} test, "
            f"outliers={config.remove_outliers}, features={config.feature_engineering}"
        )

    # Different model configurations
    print("\nModel Configuration Options:")

    model_configs = [
        ModelConfig("xgboost", param_set="best"),
        ModelConfig("lightgbm", param_set="best", use_feature_selection=True),
        ModelConfig("knn", param_set="best", override_params={"n_neighbors": 7}),
    ]

    for i, config in enumerate(model_configs, 1):
        print(
            f"   Config {i}: {config.model_type}, "
            f"feature_selection={config.use_feature_selection}, "
            f"{len(config.params)} params"
        )

    # Different MLflow configurations
    print("\nMLflow Configuration Options:")

    mlflow_configs = [
        MLflowConfig(experiment_type="hyperparameter_tuning", autolog_config="minimal"),
        MLflowConfig(experiment_type="model_comparison", autolog_config="standard"),
        MLflowConfig(experiment_type="data_pipeline", autolog_config="detailed"),
    ]

    for i, config in enumerate(mlflow_configs, 1):
        print(
            f"   Config {i}: {config.experiment_type}, "
            f"autolog={config.autolog_config['silent']}"
        )


if __name__ == "__main__":
    print("ML Experiments - Comprehensive Model Comparison Demo")
    print("This demo shows how to test and compare KNN, XGBoost, and LightGBM models")
    print("using the modular structure")

    # Run the comprehensive model comparison
    run_comprehensive_model_comparison()

    print("\n" + "=" * 80)
    print("KEY BENEFITS OF THE MODULAR STRUCTURE:")
    print("=" * 80)
    print("   - Single data preparation for all models")
    print("   - Consistent evaluation across all models")
    print("   - Centralized configuration management")
    print("   - Comprehensive performance comparison")
    print("   - Automated MLflow tracking for each model")
    print("   - Production-ready code organization")
    print("   - Easy to extend with new models")
    print("   - Type-safe with proper annotations")
    print("=" * 80)
