"""
Model evaluation and metrics calculation utilities.

This module consolidates all evaluation-related functionality that was duplicated
across multiple experiment scripts.
"""

from typing import Any, Dict, Optional

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
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


def calculate_overfitting_metrics(
    train_metrics: Dict[str, float], test_metrics: Dict[str, float]
) -> Dict[str, float]:
    """Calculate overfitting indicators.

    Args:
        train_metrics: Training set metrics
        test_metrics: Test set metrics

    Returns:
        Dictionary with overfitting ratios
    """
    overfitting_metrics = {}

    # MAE overfitting ratio (test/train - higher is worse)
    if train_metrics["mae"] > 0:
        overfitting_metrics["overfitting_ratio_mae"] = (
            test_metrics["mae"] / train_metrics["mae"]
        )

    # R² overfitting ratio (train/test - higher is worse)
    if test_metrics["r2"] > 0:
        overfitting_metrics["overfitting_ratio_r2"] = (
            train_metrics["r2"] / test_metrics["r2"]
        )

    # Performance gap metrics
    overfitting_metrics["r2_gap"] = train_metrics["r2"] - test_metrics["r2"]
    overfitting_metrics["mae_gap"] = test_metrics["mae"] - train_metrics["mae"]
    overfitting_metrics["rmse_gap"] = test_metrics["rmse"] - train_metrics["rmse"]

    return overfitting_metrics


def evaluate_model_performance(
    model,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Comprehensive model evaluation.

    Args:
        model: Trained model with predict method
        x_train: Training features
        x_test: Test features
        y_train: Training targets
        y_test: Test targets

    Returns:
        Dictionary with comprehensive evaluation results
    """
    # Make predictions
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Calculate metrics
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    overfitting_metrics = calculate_overfitting_metrics(train_metrics, test_metrics)

    # Combine all metrics
    evaluation_results = {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "overfitting_metrics": overfitting_metrics,
        "predictions": {
            "train_predictions": y_train_pred,
            "test_predictions": y_test_pred,
        },
    }

    return evaluation_results


def format_metrics_for_display(
    metrics: Dict[str, float], prefix: str = ""
) -> Dict[str, str]:
    """Format metrics for human-readable display.

    Args:
        metrics: Dictionary of metric values
        prefix: Prefix to add to metric names

    Returns:
        Dictionary with formatted metric strings
    """
    formatted = {}

    for metric_name, value in metrics.items():
        key = f"{prefix}{metric_name}" if prefix else metric_name

        if metric_name in ["mae", "mse", "rmse"]:
            # Format monetary values
            formatted[key] = f"${value:,.2f}"
        elif metric_name in ["r2"]:
            # Format as percentage or decimal
            formatted[key] = f"{value:.4f}"
        elif "ratio" in metric_name:
            # Format ratios
            formatted[key] = f"{value:.4f}"
        elif "gap" in metric_name:
            # Format gaps
            if metric_name == "r2_gap":
                formatted[key] = f"{value:.4f}"
            else:
                formatted[key] = f"${value:,.2f}"
        else:
            # Default formatting
            formatted[key] = f"{value:.4f}"

    return formatted


def get_model_performance_summary(evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
    """Get a summary of model performance.

    Args:
        evaluation_results: Results from evaluate_model_performance

    Returns:
        Dictionary with performance summary
    """
    train_metrics = evaluation_results["train_metrics"]
    test_metrics = evaluation_results["test_metrics"]
    overfitting_metrics = evaluation_results["overfitting_metrics"]

    # Determine performance level
    test_r2 = test_metrics["r2"]
    if test_r2 >= 0.9:
        performance_level = "Excellent"
    elif test_r2 >= 0.8:
        performance_level = "Good"
    elif test_r2 >= 0.7:
        performance_level = "Fair"
    elif test_r2 >= 0.5:
        performance_level = "Poor"
    else:
        performance_level = "Very Poor"

    # Determine overfitting level
    r2_ratio = overfitting_metrics.get("overfitting_ratio_r2", 1.0)
    if r2_ratio <= 1.1:
        overfitting_level = "Minimal"
    elif r2_ratio <= 1.3:
        overfitting_level = "Moderate"
    elif r2_ratio <= 1.5:
        overfitting_level = "High"
    else:
        overfitting_level = "Severe"

    return {
        "performance_level": performance_level,
        "overfitting_level": overfitting_level,
        "test_r2": test_r2,
        "test_mae": test_metrics["mae"],
        "overfitting_ratio": r2_ratio,
        "formatted_metrics": {
            "train": format_metrics_for_display(train_metrics, "train_"),
            "test": format_metrics_for_display(test_metrics, "test_"),
            "overfitting": format_metrics_for_display(overfitting_metrics),
        },
    }


class ModelEvaluator:
    """Encapsulates model evaluation logic for experiments."""

    def __init__(self, metrics_to_track: Optional[list] = None):
        """Initialize model evaluator.

        Args:
            metrics_to_track: List of metrics to track (default: all standard metrics)
        """
        self.metrics_to_track = metrics_to_track or ["mae", "mse", "rmse", "r2"]
        self.evaluation_history = []

    def evaluate(
        self,
        model,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        model_name: str = "model",
    ) -> Dict[str, Any]:
        """Evaluate a model and store results.

        Args:
            model: Trained model
            x_train: Training features
            x_test: Test features
            y_train: Training targets
            y_test: Test targets
            model_name: Name for this model evaluation

        Returns:
            Evaluation results dictionary
        """
        results = evaluate_model_performance(model, x_train, x_test, y_train, y_test)
        results["model_name"] = model_name
        results["summary"] = get_model_performance_summary(results)

        # Log metrics to MLflow if we're in an active run
        if mlflow.active_run():
            # Log train metrics
            for metric_name, value in results["train_metrics"].items():
                mlflow.log_metric(f"train_{metric_name}", value)

            # Log test metrics
            for metric_name, value in results["test_metrics"].items():
                mlflow.log_metric(f"test_{metric_name}", value)

            # Log overfitting metrics
            for metric_name, value in results["overfitting_metrics"].items():
                mlflow.log_metric(metric_name, value)

        # Store in history
        self.evaluation_history.append(results)

        return results

    def get_best_model(
        self, metric: str = "test_r2", higher_is_better: bool = True
    ) -> Dict[str, Any]:
        """Get the best model based on specified metric.

        Args:
            metric: Metric to use for comparison
            higher_is_better: Whether higher values are better for this metric

        Returns:
            Best model evaluation results
        """
        if not self.evaluation_history:
            raise ValueError("No models have been evaluated yet")

        # Extract metric values
        metric_values = []
        for result in self.evaluation_history:
            if metric.startswith("train_"):
                value = result["train_metrics"][metric.replace("train_", "")]
            elif metric.startswith("test_"):
                value = result["test_metrics"][metric.replace("test_", "")]
            elif metric in result["overfitting_metrics"]:
                value = result["overfitting_metrics"][metric]
            else:
                raise ValueError(f"Unknown metric: {metric}")
            metric_values.append(value)

        # Find best index
        if higher_is_better:
            best_idx = np.argmax(metric_values)
        else:
            best_idx = np.argmin(metric_values)

        return self.evaluation_history[best_idx]

    def _performance_to_score(self, performance_level: str) -> float:
        """Convert performance level to numeric score.

        Args:
            performance_level: Performance level string

        Returns:
            Numeric score (higher is better)
        """
        score_map = {
            "Excellent": 5.0,
            "Good": 4.0,
            "Fair": 3.0,
            "Poor": 2.0,
            "Very Poor": 1.0,
        }
        return score_map.get(performance_level, 0.0)

    def _overfitting_to_score(self, overfitting_level: str) -> float:
        """Convert overfitting level to numeric score.

        Args:
            overfitting_level: Overfitting level string

        Returns:
            Numeric score (higher is better, meaning less overfitting)
        """
        score_map = {"Minimal": 4.0, "Moderate": 3.0, "High": 2.0, "Severe": 1.0}
        return score_map.get(overfitting_level, 0.0)

    def evaluate_with_cv(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        cv_config,
        model_name: str = "model",
    ) -> Dict[str, Any]:
        """Evaluate a model using cross-validation.

        Args:
            model: Model or pipeline to evaluate
            X: Features
            y: Target variable
            cv_config: CrossValidationConfig instance
            model_name: Name for this model evaluation

        Returns:
            Cross-validation results dictionary
        """
        # Create KFold splitter
        cv = KFold(
            n_splits=cv_config.n_splits,
            shuffle=cv_config.shuffle,
            random_state=cv_config.random_state,
        )

        # Run cross-validation
        cv_results = cross_validate(
            estimator=model,
            X=X,
            y=y,
            cv=cv,
            scoring=cv_config.scoring_metrics,
            return_train_score=True,
            n_jobs=-1,
        )

        # Convert negative scores to positive (sklearn convention)
        processed_results = {}
        for metric_name, scores in cv_results.items():
            if metric_name.startswith("test_neg_") or metric_name.startswith(
                "train_neg_"
            ):
                # Remove 'neg_' prefix and convert to positive
                new_name = metric_name.replace("neg_", "")
                processed_results[new_name] = -scores
            else:
                processed_results[metric_name] = scores

        # Calculate statistics for each metric
        cv_statistics = {}
        for metric_name, scores in processed_results.items():
            if metric_name in ["fit_time", "score_time"]:
                continue

            cv_statistics[metric_name] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                "scores": scores.tolist(),  # Individual fold scores
            }

        # Prepare results
        results = {
            "model_name": model_name,
            "cv_config": cv_config.to_dict(),
            "cv_statistics": cv_statistics,
            "data_info": {
                "n_samples": len(X),
                "n_features": X.shape[1],
                "n_folds": cv_config.n_splits,
            },
            "timing": {
                "mean_fit_time": np.mean(processed_results.get("fit_time", [0])),
                "mean_score_time": np.mean(processed_results.get("score_time", [0])),
            },
        }

        # Log to MLflow if active run
        if mlflow.active_run():
            self._log_cv_results_to_mlflow(results)

        # Store in history
        self.evaluation_history.append(results)

        return results

    def _log_cv_results_to_mlflow(self, results: Dict[str, Any]):
        """Log cross-validation results to MLflow.

        Args:
            results: CV results dictionary
        """
        try:
            # Log CV configuration
            mlflow.log_params(results["cv_config"])

            # Log statistics for each metric
            for metric_name, stats in results["cv_statistics"].items():
                mlflow.log_metrics(
                    {
                        f"cv_{metric_name}_mean": stats["mean"],
                        f"cv_{metric_name}_std": stats["std"],
                        f"cv_{metric_name}_min": stats["min"],
                        f"cv_{metric_name}_max": stats["max"],
                    }
                )

            # Log timing and data info
            mlflow.log_metrics(results["timing"])
            mlflow.log_params(results["data_info"])

        except Exception as e:
            print(f"Warning: Failed to log CV results to MLflow: {e}")

    def compare_models(self) -> pd.DataFrame:
        """Compare all evaluated models.

        Returns:
            DataFrame with model comparison
        """
        if not self.evaluation_history:
            return pd.DataFrame()

        comparison_data = []
        for result in self.evaluation_history:
            row = {
                "model_name": result["model_name"],
            }

            # Handle regular evaluation results
            if "summary" in result:
                row.update(
                    {
                        "performance_level": result["summary"]["performance_level"],
                        "overfitting_level": result["summary"]["overfitting_level"],
                    }
                )

                # Add train metrics
                for metric, value in result["train_metrics"].items():
                    row[f"train_{metric}"] = value

                # Add test metrics
                for metric, value in result["test_metrics"].items():
                    row[f"test_{metric}"] = value

                # Add overfitting metrics
                for metric, value in result["overfitting_metrics"].items():
                    row[metric] = value

            # Handle CV results
            elif "cv_statistics" in result:
                for metric_name, stats in result["cv_statistics"].items():
                    row[f"cv_{metric_name}_mean"] = stats["mean"]
                    row[f"cv_{metric_name}_std"] = stats["std"]

            comparison_data.append(row)

        return pd.DataFrame(comparison_data)
