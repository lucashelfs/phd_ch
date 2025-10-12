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
from sklearn import model_selection
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature
import xgboost as xgb

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

# Hyperparameter Grid for XGBoost
PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 6, 9],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0],
    "reg_alpha": [0, 0.1, 1.0],
    "reg_lambda": [1, 1.5, 2.0],
}


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


def generate_param_combinations(
    param_grid: Dict[str, List[Any]], max_combinations: int = 50
) -> List[Dict[str, Any]]:
    """Generate parameter combinations from grid, limiting to max_combinations.

    Args:
        param_grid: Dictionary of parameter names and their possible values
        max_combinations: Maximum number of combinations to generate

    Returns:
        List of parameter dictionaries
    """
    # Get all possible combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combinations = list(product(*values))

    # Limit combinations if too many
    if len(all_combinations) > max_combinations:
        # Use numpy random sampling for reproducibility
        np.random.seed(42)
        selected_indices = np.random.choice(
            len(all_combinations), max_combinations, replace=False
        )
        selected_combinations = [all_combinations[i] for i in selected_indices]
    else:
        selected_combinations = all_combinations

    # Convert to list of dictionaries
    param_combinations = []
    for combination in selected_combinations:
        param_dict = dict(zip(keys, combination))
        param_combinations.append(param_dict)

    return param_combinations


def train_and_evaluate_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_params: Dict[str, Any],
    scaler_params: Dict[str, Any],
    combination_idx: int,
    total_combinations: int,
) -> Dict[str, Any]:
    """Train and evaluate a single model configuration.

    Args:
        x_train, x_test, y_train, y_test: Train/test splits
        model_params: XGBoost parameters
        scaler_params: Scaler parameters
        combination_idx: Current combination index
        total_combinations: Total number of combinations

    Returns:
        Dictionary with model performance metrics
    """

    # Create run name with key parameters
    run_name = f"XGBoost_Tuning_{combination_idx + 1:02d}_n{model_params['n_estimators']}_d{model_params['max_depth']}_lr{model_params['learning_rate']}"

    with mlflow.start_run(run_name=run_name) as run:
        print(f"[{combination_idx + 1}/{total_combinations}] Training: {run_name}")

        # Log dataset information (only for first run to avoid duplication)
        if combination_idx == 0:
            log_data_info(x_train, y_train)

        # Log model parameters
        mlflow.log_params(
            {
                **{f"xgb_{k}": v for k, v in model_params.items()},
                **{f"scaler_{k}": v for k, v in scaler_params.items()},
                "test_size": 0.2,
                "random_state": 42,
                "combination_index": combination_idx + 1,
                "total_combinations": total_combinations,
            }
        )

        # Create and train model pipeline
        model = pipeline.make_pipeline(
            preprocessing.RobustScaler(**scaler_params),
            xgb.XGBRegressor(**model_params),
        )

        model.fit(x_train, y_train)

        # Make predictions
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        # Calculate metrics
        train_metrics = calculate_metrics(y_train, y_train_pred)
        test_metrics = calculate_metrics(y_test, y_test_pred)

        print(f"   Test R²: {test_metrics['r2']:.4f}, MAE: ${test_metrics['mae']:,.0f}")

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

        # Set tags for better organization
        mlflow.set_tags(
            {
                "model_type": "XGBRegressor_Tuned",
                "preprocessing": "RobustScaler",
                "problem_type": "regression",
                "domain": "real_estate",
                "data_source": "kc_house_data",
                "author": "MLflow Hyperparameter Tuning",
                "version": "1.0",
                "tuning_run": "true",
            }
        )

        # Return results for summary
        return {
            "run_id": run.info.run_id,
            "params": model_params.copy(),
            "test_r2": test_metrics["r2"],
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "overfitting_ratio_r2": train_metrics["r2"] / test_metrics["r2"],
        }


def main():
    """Load data, perform hyperparameter tuning for XGBoost with MLflow tracking."""

    # Set MLflow tracking URI and experiment
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Enable autologging for XGBoost (but we'll do manual logging for better control)
    mlflow.xgboost.autolog(
        log_input_examples=False, log_model_signatures=False, silent=True
    )

    print("Starting XGBoost Hyperparameter Tuning")
    print("=" * 60)

    print(f"Loading data from {SALES_PATH} and {DEMOGRAPHICS_PATH}")
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)

    print(f"Dataset loaded: {len(x)} samples, {len(x.columns)} features")
    print(f"Target statistics: mean=${y.mean():.2f}, std=${y.std():.2f}")

    # Split data for proper evaluation
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # Scaler parameters (keep consistent)
    scaler_params = {
        "quantile_range": (25.0, 75.0),
        "with_centering": True,
        "with_scaling": True,
    }

    # Generate parameter combinations
    param_combinations = generate_param_combinations(PARAM_GRID, max_combinations=50)
    total_combinations = len(param_combinations)

    print(f"\nTesting {total_combinations} parameter combinations")
    print("=" * 60)

    # Store results for summary
    results = []

    # Train models with different parameter combinations
    for idx, params in enumerate(param_combinations):
        # Add fixed parameters
        model_params = {
            **params,
            "random_state": 42,
            "n_jobs": -1,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
        }

        try:
            result = train_and_evaluate_model(
                x_train,
                x_test,
                y_train,
                y_test,
                model_params,
                scaler_params,
                idx,
                total_combinations,
            )
            results.append(result)

        except Exception as e:
            print(f"   Error with combination {idx + 1}: {str(e)}")
            continue

    # Summary of results
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING SUMMARY")
    print("=" * 60)

    if results:
        # Sort by test R²
        results_sorted = sorted(results, key=lambda x: x["test_r2"], reverse=True)

        print(f"\nTop 5 Configurations (by Test R²):")
        print("-" * 60)

        for i, result in enumerate(results_sorted[:5]):
            print(
                f"\n{i + 1}. Test R²: {result['test_r2']:.4f} | MAE: ${result['test_mae']:,.0f}"
            )
            print(
                f"   Parameters: n_est={result['params']['n_estimators']}, "
                f"depth={result['params']['max_depth']}, "
                f"lr={result['params']['learning_rate']}, "
                f"reg_a={result['params']['reg_alpha']}"
            )
            print(f"   Run ID: {result['run_id'][:8]}...")

        # Best model summary
        best_result = results_sorted[0]
        print(f"\nBEST CONFIGURATION:")
        print(f"   Test R²: {best_result['test_r2']:.4f}")
        print(f"   Test MAE: ${best_result['test_mae']:,.2f}")
        print(f"   Test RMSE: ${best_result['test_rmse']:,.2f}")
        print(f"   Overfitting Ratio: {best_result['overfitting_ratio_r2']:.4f}")
        print(f"   Run ID: {best_result['run_id']}")

        # Performance statistics
        r2_scores = [r["test_r2"] for r in results]
        mae_scores = [r["test_mae"] for r in results]

        print(f"\nPERFORMANCE STATISTICS:")
        print(f"   R² Range: {min(r2_scores):.4f} - {max(r2_scores):.4f}")
        print(f"   R² Mean: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
        print(f"   MAE Range: ${min(mae_scores):,.0f} - ${max(mae_scores):,.0f}")
        print(f"   MAE Mean: ${np.mean(mae_scores):,.0f} ± ${np.std(mae_scores):,.0f}")

        # Save summary
        summary = {
            "tuning_type": "XGBoost Hyperparameter Tuning",
            "total_combinations_tested": len(results),
            "best_result": best_result,
            "top_5_results": results_sorted[:5],
            "performance_stats": {
                "r2_range": [min(r2_scores), max(r2_scores)],
                "r2_mean": np.mean(r2_scores),
                "r2_std": np.std(r2_scores),
                "mae_range": [min(mae_scores), max(mae_scores)],
                "mae_mean": np.mean(mae_scores),
                "mae_std": np.std(mae_scores),
            },
        }

        with open("xgboost_tuning_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\nSummary saved to: xgboost_tuning_summary.json")

    else:
        print("No successful runs completed!")

    print(f"\nView detailed results in MLflow UI:")
    print(f"   1. Run: mlflow ui --backend-store-uri file:./mlruns")
    print(f"   2. Open: http://localhost:5000")
    print(f"   3. Navigate to '{EXPERIMENT_NAME}' experiment")
    print(f"   4. Filter by tag 'tuning_run=true' to see only tuning runs")


if __name__ == "__main__":
    main()
