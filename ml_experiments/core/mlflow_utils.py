"""
Simplified MLflow tracking utilities.

This module provides clean, direct MLflow functionality following the pattern
from create_champion_model_with_artifacts.py to avoid logging "none values"
and unnecessary complexity.
"""

from typing import Any, Dict, Optional

import mlflow
import mlflow.sklearn
import pandas as pd


def setup_mlflow(
    tracking_uri: str = "file:./mlruns",
    experiment_name: str = "ML Experiments",
) -> str:
    """Setup MLflow tracking configuration.

    Args:
        tracking_uri: MLflow tracking URI
        experiment_name: Name of the experiment

    Returns:
        Experiment ID
    """
    # Set tracking URI
    mlflow.set_tracking_uri(tracking_uri)

    # Set or create experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
    except Exception:
        experiment_id = mlflow.create_experiment(experiment_name)

    mlflow.set_experiment(experiment_name)
    return experiment_id


def create_run_name(
    model_type: str,
    experiment_idx: int,
    key_params: Optional[Dict[str, Any]] = None,
) -> str:
    """Create a descriptive run name.

    Args:
        model_type: Type of model
        experiment_idx: Experiment index
        key_params: Key parameters to include in name

    Returns:
        Formatted run name
    """
    base_name = f"{model_type.upper()}_{experiment_idx:03d}"

    if key_params:
        param_str = "_".join([f"{k}{v}" for k, v in key_params.items()])
        return f"{base_name}_{param_str}"

    return base_name


def get_experiment_results(
    experiment_name: str,
    tracking_uri: str = "file:./mlruns",
    filter_string: str = "status = 'FINISHED'",
    order_by: Optional[list] = None,
) -> pd.DataFrame:
    """Get results from an MLflow experiment.

    Args:
        experiment_name: Name of the experiment
        tracking_uri: MLflow tracking URI
        filter_string: Filter for runs
        order_by: Columns to order by

    Returns:
        DataFrame with experiment results
    """
    mlflow.set_tracking_uri(tracking_uri)

    # Get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return pd.DataFrame()

    # Get runs
    order_by = order_by or ["metrics.test_r2 DESC"]
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        order_by=order_by,
    )

    # Ensure we return a DataFrame
    if isinstance(runs, list):
        return pd.DataFrame()

    return runs


def compare_experiment_runs(
    experiment_name: str,
    tracking_uri: str = "file:./mlruns",
    metrics: Optional[list] = None,
) -> pd.DataFrame:
    """Compare runs from an experiment.

    Args:
        experiment_name: Name of the experiment
        tracking_uri: MLflow tracking URI
        metrics: Metrics to include in comparison

    Returns:
        DataFrame with run comparison
    """
    runs_df = get_experiment_results(experiment_name, tracking_uri)

    if runs_df.empty:
        return pd.DataFrame()

    # Select relevant columns
    comparison_cols = [
        "run_id",
        "tags.mlflow.runName",
        "tags.model_type",
        "start_time",
        "status",
    ]

    # Add metric columns
    metrics = metrics or [
        "test_r2",
        "test_mae",
        "test_rmse",
        "train_r2",
        "overfitting_ratio_r2",
    ]
    for metric in metrics:
        metric_col = f"metrics.{metric}"
        if metric_col in runs_df.columns:
            comparison_cols.append(metric_col)

    # Filter to available columns
    available_cols = [col for col in comparison_cols if col in runs_df.columns]
    comparison_df = runs_df[available_cols].copy()

    # Clean up column names
    comparison_df.columns = [
        col.replace("metrics.", "").replace("tags.", "")
        for col in comparison_df.columns
    ]

    return comparison_df


def get_best_run(
    experiment_name: str,
    metric: str = "test_r2",
    tracking_uri: str = "file:./mlruns",
    higher_is_better: bool = True,
) -> Optional[Dict[str, Any]]:
    """Get the best run from an experiment.

    Args:
        experiment_name: Name of the experiment
        metric: Metric to optimize
        tracking_uri: MLflow tracking URI
        higher_is_better: Whether higher values are better

    Returns:
        Best run information or None
    """
    runs_df = get_experiment_results(experiment_name, tracking_uri)

    if runs_df.empty:
        return None

    metric_col = f"metrics.{metric}"
    if metric_col not in runs_df.columns:
        return None

    # Find best run
    if higher_is_better:
        best_idx = runs_df[metric_col].idxmax()
    else:
        best_idx = runs_df[metric_col].idxmin()

    best_run = runs_df.loc[best_idx]

    return {
        "run_id": best_run["run_id"],
        "run_name": best_run.get("tags.mlflow.runName", "Unknown"),
        "metric_value": best_run[metric_col],
        "start_time": best_run["start_time"],
        "model_type": best_run.get("tags.model_type", "Unknown"),
    }
