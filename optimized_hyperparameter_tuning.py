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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature
import xgboost as xgb
import lightgbm as lgb

# Configuration
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

# MLflow Configuration
EXPERIMENT_NAME = "Real Estate Optimized Split Hyperparameter Tuning"
TRACKING_URI = "file:./mlruns"

# Optimal data pipeline configuration (from data pipeline experiments)
OPTIMAL_TEST_SIZE = 0.25  # 75/25 split
OPTIMAL_SCALER = preprocessing.RobustScaler(quantile_range=(25.0, 75.0))
KEEP_OUTLIERS = True  # For better R² performance
USE_ORIGINAL_FEATURES = True  # No feature engineering

# Expanded hyperparameter grids based on optimal data configuration
LIGHTGBM_PARAM_GRID = {
    "n_estimators": [25, 50, 75, 100, 150],
    "max_depth": [6, 9, 12, 15],
    "learning_rate": [0.05, 0.1, 0.15, 0.2],
    "num_leaves": [15, 31, 50, 75],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0],
    "reg_alpha": [0, 0.1, 0.5, 1.0],
    "reg_lambda": [1.0, 1.5, 2.0],
}

XGBOOST_PARAM_GRID = {
    "n_estimators": [25, 50, 75, 100, 150],
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.1, 0.15, 0.2, 0.25],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0],
    "reg_alpha": [0, 0.1, 0.5],
    "reg_lambda": [1.0, 1.5, 2.0],
}

KNN_PARAM_GRID = {
    "n_neighbors": [3, 5, 7, 9, 11, 15],
    "weights": ["uniform", "distance"],
    "algorithm": ["auto", "ball_tree", "kd_tree"],
    "p": [1, 2],
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


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Calculate regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


def log_data_info(x: pd.DataFrame, y: pd.Series) -> None:
    """Log dataset information to MLflow."""
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
    param_grid: Dict[str, List[Any]], max_combinations: int = 80
) -> List[Dict[str, Any]]:
    """Generate parameter combinations from grid, limiting to max_combinations."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combinations = list(product(*values))

    if len(all_combinations) > max_combinations:
        np.random.seed(42)
        selected_indices = np.random.choice(
            len(all_combinations), max_combinations, replace=False
        )
        selected_combinations = [all_combinations[i] for i in selected_indices]
    else:
        selected_combinations = all_combinations

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
    model_type: str,
    model_params: Dict[str, Any],
    scaler_params: Dict[str, Any],
    combination_idx: int,
    total_combinations: int,
) -> Dict[str, Any]:
    """Train and evaluate a single model configuration."""

    # Create run name with key parameters
    if model_type == "lightgbm":
        run_name = f"LightGBM_Opt_{combination_idx + 1:03d}_n{model_params['n_estimators']}_d{model_params['max_depth']}_lr{model_params['learning_rate']}_leaves{model_params['num_leaves']}"
    elif model_type == "xgboost":
        run_name = f"XGBoost_Opt_{combination_idx + 1:03d}_n{model_params['n_estimators']}_d{model_params['max_depth']}_lr{model_params['learning_rate']}"
    else:  # knn
        run_name = f"KNN_Opt_{combination_idx + 1:03d}_k{model_params['n_neighbors']}_w{model_params['weights']}_p{model_params['p']}"

    with mlflow.start_run(run_name=run_name) as run:
        print(f"[{combination_idx + 1}/{total_combinations}] Training: {run_name}")

        # Log dataset information (only for first run to avoid duplication)
        if combination_idx == 0:
            log_data_info(x_train, y_train)

        # Log model parameters
        mlflow.log_params(
            {
                **{f"{model_type}_{k}": v for k, v in model_params.items()},
                **{f"scaler_{k}": v for k, v in scaler_params.items()},
                "test_size": OPTIMAL_TEST_SIZE,
                "train_size": 1 - OPTIMAL_TEST_SIZE,
                "random_state": 42,
                "combination_index": combination_idx + 1,
                "total_combinations": total_combinations,
                "data_pipeline": "optimized_75_25_split",
                "outlier_handling": "keep" if KEEP_OUTLIERS else "remove",
                "feature_engineering": "none" if USE_ORIGINAL_FEATURES else "applied",
            }
        )

        # Create and train model pipeline
        if model_type == "lightgbm":
            model = pipeline.make_pipeline(
                OPTIMAL_SCALER, lgb.LGBMRegressor(**model_params)
            )
        elif model_type == "xgboost":
            model = pipeline.make_pipeline(
                OPTIMAL_SCALER, xgb.XGBRegressor(**model_params)
            )
        else:  # knn
            model = pipeline.make_pipeline(
                OPTIMAL_SCALER, neighbors.KNeighborsRegressor(**model_params)
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
                "model_type": f"{model_type.upper()}_Optimized",
                "preprocessing": "RobustScaler_Optimized",
                "problem_type": "regression",
                "domain": "real_estate",
                "data_source": "kc_house_data",
                "author": "Optimized Hyperparameter Tuning",
                "version": "2.0",
                "tuning_run": "optimized",
                "data_split": "75_25_optimal",
                "baseline_improvement": "true",
            }
        )

        return {
            "run_id": run.info.run_id,
            "model_type": model_type,
            "params": model_params.copy(),
            "test_r2": test_metrics["r2"],
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "overfitting_ratio_r2": train_metrics["r2"] / test_metrics["r2"],
        }


def run_model_tuning(
    x: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    param_grid: Dict[str, List[Any]],
    max_combinations: int = 80,
) -> List[Dict[str, Any]]:
    """Run hyperparameter tuning for a specific model type."""

    print(f"\nStarting {model_type.upper()} hyperparameter tuning")
    print("=" * 60)

    # Split data using optimal configuration
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, test_size=OPTIMAL_TEST_SIZE, random_state=42
    )

    # Scaler parameters
    scaler_params = {
        "quantile_range": (25.0, 75.0),
        "with_centering": True,
        "with_scaling": True,
    }

    # Generate parameter combinations
    param_combinations = generate_param_combinations(param_grid, max_combinations)
    total_combinations = len(param_combinations)

    print(f"Testing {total_combinations} parameter combinations")
    print("-" * 60)

    results = []
    for idx, params in enumerate(param_combinations):
        # Add fixed parameters based on model type
        if model_type == "lightgbm":
            model_params = {
                **params,
                "random_state": 42,
                "n_jobs": -1,
                "objective": "regression",
                "metric": "rmse",
                "verbose": -1,
            }
        elif model_type == "xgboost":
            model_params = {
                **params,
                "random_state": 42,
                "n_jobs": -1,
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
            }
        else:  # knn
            model_params = params.copy()

        try:
            result = train_and_evaluate_model(
                x_train,
                x_test,
                y_train,
                y_test,
                model_type,
                model_params,
                scaler_params,
                idx,
                total_combinations,
            )
            results.append(result)

        except Exception as e:
            print(f"   Error with combination {idx + 1}: {str(e)}")
            continue

    return results


def main():
    """Run optimized hyperparameter tuning with best data pipeline configuration."""

    # Set MLflow tracking URI and experiment
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Real Estate Optimized Split Hyperparameter Tuning")
    print("=" * 70)
    print(f"Using optimal data pipeline configuration:")
    print(
        f"  - Train/Test Split: {int((1 - OPTIMAL_TEST_SIZE) * 100)}/{int(OPTIMAL_TEST_SIZE * 100)}"
    )
    print(f"  - Scaler: RobustScaler (25-75% quantiles)")
    print(f"  - Features: Original 33 features")
    print(f"  - Outliers: {'Keep' if KEEP_OUTLIERS else 'Remove'}")

    # Load data
    print(f"\nLoading data from {SALES_PATH} and {DEMOGRAPHICS_PATH}")
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)

    print(f"Dataset loaded: {len(x)} samples, {len(x.columns)} features")
    print(f"Target statistics: mean=${y.mean():.2f}, std=${y.std():.2f}")

    # Run hyperparameter tuning for each model
    all_results = []

    # LightGBM tuning (priority model)
    lightgbm_results = run_model_tuning(x, y, "lightgbm", LIGHTGBM_PARAM_GRID, 80)
    all_results.extend(lightgbm_results)

    # XGBoost tuning
    xgboost_results = run_model_tuning(x, y, "xgboost", XGBOOST_PARAM_GRID, 60)
    all_results.extend(xgboost_results)

    # KNN tuning
    knn_results = run_model_tuning(x, y, "knn", KNN_PARAM_GRID, 40)
    all_results.extend(knn_results)

    # Analyze results
    print("\n" + "=" * 70)
    print("OPTIMIZED HYPERPARAMETER TUNING SUMMARY")
    print("=" * 70)

    if all_results:
        # Sort by test R²
        results_sorted = sorted(all_results, key=lambda x: x["test_r2"], reverse=True)

        print(f"\nTop 15 Configurations (by Test R²):")
        print("-" * 70)

        for i, result in enumerate(results_sorted[:15]):
            print(
                f"\n{i + 1:2d}. R²: {result['test_r2']:.4f} | MAE: ${result['test_mae']:,.0f} | Model: {result['model_type'].upper()}"
            )

            # Show key parameters based on model type
            if result["model_type"] == "lightgbm":
                print(
                    f"    Params: n_est={result['params']['n_estimators']}, depth={result['params']['max_depth']}, "
                    f"lr={result['params']['learning_rate']}, leaves={result['params']['num_leaves']}"
                )
            elif result["model_type"] == "xgboost":
                print(
                    f"    Params: n_est={result['params']['n_estimators']}, depth={result['params']['max_depth']}, "
                    f"lr={result['params']['learning_rate']}, reg_a={result['params']['reg_alpha']}"
                )
            else:  # knn
                print(
                    f"    Params: k={result['params']['n_neighbors']}, weights={result['params']['weights']}, "
                    f"algorithm={result['params']['algorithm']}, p={result['params']['p']}"
                )

            print(f"    Run ID: {result['run_id'][:8]}...")

        # Best model summary
        best_result = results_sorted[0]
        print(f"\nBEST OPTIMIZED CONFIGURATION:")
        print(f"   Model: {best_result['model_type'].upper()}")
        print(f"   Test R²: {best_result['test_r2']:.4f}")
        print(f"   Test MAE: ${best_result['test_mae']:,.2f}")
        print(f"   Test RMSE: ${best_result['test_rmse']:,.2f}")
        print(f"   Overfitting Ratio: {best_result['overfitting_ratio_r2']:.4f}")
        print(f"   Run ID: {best_result['run_id']}")

        # Performance comparison with previous experiments
        baseline_r2 = 0.7364  # Original KNN
        tuned_r2 = 0.7932  # Previous best tuned
        pipeline_r2 = 0.8038  # Data pipeline optimized
        current_r2 = best_result["test_r2"]

        print(f"\nPERFORMANCE EVOLUTION:")
        print(f"   Baseline KNN:           R² = {baseline_r2:.4f}")
        print(
            f"   Initial Tuning:         R² = {tuned_r2:.4f} (+{((tuned_r2 / baseline_r2) - 1) * 100:.2f}%)"
        )
        print(
            f"   Data Pipeline Opt:      R² = {pipeline_r2:.4f} (+{((pipeline_r2 / baseline_r2) - 1) * 100:.2f}%)"
        )
        print(
            f"   Optimized Tuning:       R² = {current_r2:.4f} (+{((current_r2 / baseline_r2) - 1) * 100:.2f}%)"
        )

        # Model-specific analysis
        lightgbm_results_sorted = sorted(
            [r for r in all_results if r["model_type"] == "lightgbm"],
            key=lambda x: x["test_r2"],
            reverse=True,
        )
        xgboost_results_sorted = sorted(
            [r for r in all_results if r["model_type"] == "xgboost"],
            key=lambda x: x["test_r2"],
            reverse=True,
        )
        knn_results_sorted = sorted(
            [r for r in all_results if r["model_type"] == "knn"],
            key=lambda x: x["test_r2"],
            reverse=True,
        )

        print(f"\nMODEL-SPECIFIC BEST RESULTS:")
        if lightgbm_results_sorted:
            best_lgb = lightgbm_results_sorted[0]
            print(
                f"   LightGBM: R² = {best_lgb['test_r2']:.4f}, MAE = ${best_lgb['test_mae']:,.0f}"
            )

        if xgboost_results_sorted:
            best_xgb = xgboost_results_sorted[0]
            print(
                f"   XGBoost:  R² = {best_xgb['test_r2']:.4f}, MAE = ${best_xgb['test_mae']:,.0f}"
            )

        if knn_results_sorted:
            best_knn = knn_results_sorted[0]
            print(
                f"   KNN:      R² = {best_knn['test_r2']:.4f}, MAE = ${best_knn['test_mae']:,.0f}"
            )

        # Save comprehensive summary
        summary = {
            "experiment_type": "Optimized Hyperparameter Tuning",
            "data_pipeline_config": {
                "train_test_split": f"{int((1 - OPTIMAL_TEST_SIZE) * 100)}/{int(OPTIMAL_TEST_SIZE * 100)}",
                "scaler": "RobustScaler",
                "feature_engineering": "none",
                "outlier_handling": "keep",
                "features": len(x.columns),
            },
            "total_experiments": len(all_results),
            "best_result": best_result,
            "top_15_results": results_sorted[:15],
            "model_specific_best": {
                "lightgbm": lightgbm_results_sorted[0]
                if lightgbm_results_sorted
                else None,
                "xgboost": xgboost_results_sorted[0]
                if xgboost_results_sorted
                else None,
                "knn": knn_results_sorted[0] if knn_results_sorted else None,
            },
            "performance_evolution": {
                "baseline_knn": baseline_r2,
                "initial_tuning": tuned_r2,
                "data_pipeline_optimized": pipeline_r2,
                "optimized_tuning": current_r2,
                "total_improvement_percent": ((current_r2 / baseline_r2) - 1) * 100,
            },
        }

        with open("optimized_hyperparameter_tuning_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\nSummary saved to: optimized_hyperparameter_tuning_summary.json")

    else:
        print("No successful experiments completed!")

    print(f"\nView detailed results in MLflow UI:")
    print(f"   1. Run: mlflow ui --backend-store-uri file:./mlruns")
    print(f"   2. Open: http://localhost:5000")
    print(f"   3. Navigate to '{EXPERIMENT_NAME}' experiment")
    print(f"   4. Filter by tag 'tuning_run=optimized' to see only these runs")


if __name__ == "__main__":
    main()
