"""
Model comparison script for Real Estate Price Prediction.
This script demonstrates how to compare KNN, XGBoost, and LightGBM models using MLflow.
"""

import mlflow
import pandas as pd
from typing import Dict, List

# MLflow Configuration
TRACKING_URI = "file:./mlruns"
EXPERIMENT_NAME = "Real Estate Price Prediction"


def get_model_comparison() -> pd.DataFrame:
    """Get comparison of all models in the experiment."""
    mlflow.set_tracking_uri(TRACKING_URI)

    # Get experiment
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print(f"Experiment '{EXPERIMENT_NAME}' not found!")
        return pd.DataFrame()

    # Get all runs from the experiment
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["metrics.test_r2 DESC"],
    )

    if runs.empty:
        print("No finished runs found in the experiment!")
        return pd.DataFrame()

    # Select relevant columns for comparison
    comparison_cols = [
        "run_id",
        "tags.mlflow.runName",
        "tags.model_type",
        "metrics.test_r2",
        "metrics.test_mae",
        "metrics.test_rmse",
        "metrics.train_r2",
        "metrics.train_mae",
        "metrics.overfitting_ratio_r2",
        "start_time",
    ]

    # Filter to only include columns that exist
    available_cols = [col for col in comparison_cols if col in runs.columns]
    comparison_df = runs[available_cols].copy()

    # Clean up column names
    comparison_df.columns = [
        col.replace("metrics.", "").replace("tags.", "")
        for col in comparison_df.columns
    ]

    return comparison_df


def print_model_rankings():
    """Print model rankings based on different metrics."""
    comparison_df = get_model_comparison()

    if comparison_df.empty:
        return

    print("=" * 80)
    print("REAL ESTATE PRICE PREDICTION - MODEL COMPARISON")
    print("=" * 80)

    print(f"\nFound {len(comparison_df)} models to compare:\n")

    # Display detailed comparison
    for idx, row in comparison_df.iterrows():
        model_name = row.get("mlflow.runName", "Unknown")
        model_type = row.get("model_type", "Unknown")

        print(f"🏠 {model_name}")
        print(f"   Model Type: {model_type}")
        print(f"   Test R²: {row.get('test_r2', 0):.4f}")
        print(f"   Test MAE: ${row.get('test_mae', 0):,.2f}")
        print(f"   Test RMSE: ${row.get('test_rmse', 0):,.2f}")
        print(f"   Train R²: {row.get('train_r2', 0):.4f}")
        print(f"   Overfitting Ratio: {row.get('overfitting_ratio_r2', 0):.4f}")
        print(f"   Run ID: {row['run_id'][:8]}...")
        print()

    # Rankings by different metrics
    print("📊 MODEL RANKINGS")
    print("-" * 40)

    if "test_r2" in comparison_df.columns:
        print("🥇 Best R² Score (Test Set):")
        best_r2 = comparison_df.loc[comparison_df["test_r2"].idxmax()]
        print(
            f"   {best_r2.get('mlflow.runName', 'Unknown')} - R²: {best_r2['test_r2']:.4f}"
        )

    if "test_mae" in comparison_df.columns:
        print("🥇 Lowest MAE (Test Set):")
        best_mae = comparison_df.loc[comparison_df["test_mae"].idxmin()]
        print(
            f"   {best_mae.get('mlflow.runName', 'Unknown')} - MAE: ${best_mae['test_mae']:,.2f}"
        )

    if "overfitting_ratio_r2" in comparison_df.columns:
        print("🥇 Best Generalization (Lowest Overfitting):")
        best_gen = comparison_df.loc[comparison_df["overfitting_ratio_r2"].idxmin()]
        print(
            f"   {best_gen.get('mlflow.runName', 'Unknown')} - Ratio: {best_gen['overfitting_ratio_r2']:.4f}"
        )

    print()


def get_model_summary_stats():
    """Get summary statistics across all models."""
    comparison_df = get_model_comparison()

    if comparison_df.empty:
        return

    print("📈 SUMMARY STATISTICS")
    print("-" * 40)

    numeric_cols = ["test_r2", "test_mae", "test_rmse", "train_r2"]
    available_numeric_cols = [
        col for col in numeric_cols if col in comparison_df.columns
    ]

    if available_numeric_cols:
        summary = comparison_df[available_numeric_cols].describe()
        print(summary.round(4))

    print()


def recommend_best_model():
    """Recommend the best model based on multiple criteria."""
    comparison_df = get_model_comparison()

    if comparison_df.empty:
        return

    print("🎯 MODEL RECOMMENDATION")
    print("-" * 40)

    # Score models based on multiple criteria
    scores = []

    for idx, row in comparison_df.iterrows():
        score = 0
        criteria_count = 0

        # R² score (higher is better) - weight: 40%
        if "test_r2" in row and pd.notna(row["test_r2"]):
            r2_normalized = row["test_r2"] / comparison_df["test_r2"].max()
            score += r2_normalized * 0.4
            criteria_count += 0.4

        # MAE (lower is better) - weight: 30%
        if "test_mae" in row and pd.notna(row["test_mae"]):
            mae_normalized = comparison_df["test_mae"].min() / row["test_mae"]
            score += mae_normalized * 0.3
            criteria_count += 0.3

        # Overfitting ratio (lower is better) - weight: 30%
        if "overfitting_ratio_r2" in row and pd.notna(row["overfitting_ratio_r2"]):
            # Avoid division by zero
            min_ratio = comparison_df["overfitting_ratio_r2"].min()
            if min_ratio > 0:
                ratio_normalized = min_ratio / row["overfitting_ratio_r2"]
                score += ratio_normalized * 0.3
                criteria_count += 0.3

        # Normalize score by criteria count
        if criteria_count > 0:
            score = score / criteria_count

        scores.append(score)

    comparison_df["composite_score"] = scores
    best_model = comparison_df.loc[comparison_df["composite_score"].idxmax()]

    print(f"🏆 RECOMMENDED MODEL: {best_model.get('mlflow.runName', 'Unknown')}")
    print(f"   Model Type: {best_model.get('model_type', 'Unknown')}")
    print(f"   Composite Score: {best_model['composite_score']:.4f}")
    print(f"   Test R²: {best_model.get('test_r2', 0):.4f}")
    print(f"   Test MAE: ${best_model.get('test_mae', 0):,.2f}")
    print(f"   Overfitting Ratio: {best_model.get('overfitting_ratio_r2', 0):.4f}")
    print()

    print("💡 REASONING:")
    print("   This recommendation is based on a composite score considering:")
    print("   • Test R² Score (40% weight) - Model accuracy")
    print("   • Test MAE (30% weight) - Prediction error in dollars")
    print("   • Overfitting Ratio (30% weight) - Model generalization")
    print()


def main():
    """Main comparison function."""
    print_model_rankings()
    get_model_summary_stats()
    recommend_best_model()

    print("🔗 VIEW DETAILED RESULTS:")
    print("   1. Run: mlflow ui --backend-store-uri file:./mlruns")
    print("   2. Open: http://localhost:5000")
    print("   3. Navigate to 'Real Estate Price Prediction' experiment")
    print("   4. Compare runs side-by-side using the MLflow UI")
    print()


if __name__ == "__main__":
    main()
