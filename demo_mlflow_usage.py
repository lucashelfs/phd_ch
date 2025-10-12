"""
Demo script showing how to use MLflow-tracked models for predictions.
This demonstrates how to load models from the MLflow registry and make predictions.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from typing import Dict, Any

# MLflow Configuration
TRACKING_URI = "file:./mlruns"
EXPERIMENT_NAME = "Real Estate Price Prediction"
MODEL_NAME = "real_estate_knn_model"


def load_latest_model():
    """Load the latest version of the registered model."""
    mlflow.set_tracking_uri(TRACKING_URI)

    # Load the latest version of the registered model
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/latest")
    print(f"Loaded model: {MODEL_NAME} (latest version)")

    return model


def load_model_by_version(version: int):
    """Load a specific version of the registered model."""
    mlflow.set_tracking_uri(TRACKING_URI)

    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")
    print(f"Loaded model: {MODEL_NAME} (version {version})")

    return model


def load_model_by_run_id(run_id: str):
    """Load a model from a specific MLflow run."""
    mlflow.set_tracking_uri(TRACKING_URI)

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    print(f"Loaded model from run: {run_id}")

    return model


def get_experiment_info():
    """Get information about the experiment and its runs."""
    mlflow.set_tracking_uri(TRACKING_URI)

    # Get experiment
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print(f"Experiment '{EXPERIMENT_NAME}' not found!")
        return

    print(f"Experiment: {experiment.name}")
    print(f"Experiment ID: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")

    # Get runs from the experiment
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    print(f"\nFound {len(runs)} runs:")

    for idx, run in runs.iterrows():
        print(f"\nRun {idx + 1}:")
        print(f"  Run ID: {run['run_id']}")
        print(f"  Status: {run['status']}")
        print(f"  Start Time: {run['start_time']}")
        print(f"  Test R²: {run.get('metrics.test_r2', 'N/A'):.4f}")
        print(f"  Test MAE: ${run.get('metrics.test_mae', 0):.2f}")
        print(f"  Model Type: {run.get('tags.model_type', 'N/A')}")

    return runs


def make_sample_predictions():
    """Make predictions using the latest model with sample data."""
    # Load the model
    model = load_latest_model()

    # Create sample data (using feature names from the training)
    sample_data = pd.DataFrame(
        {
            "bedrooms": [3, 4, 2, 5],
            "bathrooms": [2.0, 2.5, 1.0, 3.0],
            "sqft_living": [1800, 2500, 1200, 3200],
            "sqft_lot": [7200, 9000, 5000, 12000],
            "floors": [1.0, 2.0, 1.0, 2.0],
            "sqft_above": [1800, 1800, 1200, 2200],
            "sqft_basement": [0, 700, 0, 1000],
            # Demographics data (example values)
            "ppltn_qty": [25000, 30000, 20000, 35000],
            "urbn_ppltn_qty": [20000, 25000, 15000, 30000],
            "sbrbn_ppltn_qty": [5000, 5000, 5000, 5000],
            "farm_ppltn_qty": [0, 0, 0, 0],
            "non_farm_qty": [25000, 30000, 20000, 35000],
            "medn_hshld_incm_amt": [65000, 75000, 55000, 85000],
            "medn_incm_per_prsn_amt": [35000, 40000, 30000, 45000],
            "hous_val_amt": [400000, 500000, 300000, 600000],
            "edctn_less_than_9_qty": [500, 600, 400, 700],
            "edctn_9_12_qty": [2000, 2500, 1500, 3000],
            "edctn_high_schl_qty": [8000, 10000, 6000, 12000],
            "edctn_some_clg_qty": [6000, 7500, 4500, 9000],
            "edctn_assoc_dgre_qty": [3000, 3500, 2500, 4000],
            "edctn_bchlr_dgre_qty": [4000, 5000, 3000, 6000],
            "edctn_prfsnl_qty": [1500, 2000, 1000, 2500],
            "per_urbn": [0.8, 0.83, 0.75, 0.86],
            "per_sbrbn": [0.2, 0.17, 0.25, 0.14],
            "per_farm": [0.0, 0.0, 0.0, 0.0],
            "per_non_farm": [1.0, 1.0, 1.0, 1.0],
            "per_less_than_9": [0.02, 0.02, 0.02, 0.02],
            "per_9_to_12": [0.08, 0.08, 0.075, 0.086],
            "per_hsd": [0.32, 0.33, 0.30, 0.34],
            "per_some_clg": [0.24, 0.25, 0.225, 0.26],
            "per_assoc": [0.12, 0.12, 0.125, 0.11],
            "per_bchlr": [0.16, 0.17, 0.15, 0.17],
            "per_prfsnl": [0.06, 0.07, 0.05, 0.07],
        }
    )

    # Make predictions
    predictions = model.predict(sample_data)

    print(f"\nSample Predictions:")
    print("-" * 50)
    for i, (idx, row) in enumerate(sample_data.iterrows()):
        print(f"House {i + 1}:")
        print(f"  Bedrooms: {row['bedrooms']}, Bathrooms: {row['bathrooms']}")
        print(f"  Sqft Living: {row['sqft_living']:,}, Lot: {row['sqft_lot']:,}")
        print(f"  Predicted Price: ${predictions[i]:,.2f}")
        print()

    return predictions


def compare_model_versions():
    """Compare predictions from different model versions."""
    mlflow.set_tracking_uri(TRACKING_URI)

    try:
        # Try to load different versions
        model_v1 = load_model_by_version(1)
        model_v2 = load_model_by_version(2)

        # Create simple test data
        test_data = pd.DataFrame(
            {
                "bedrooms": [3],
                "bathrooms": [2.0],
                "sqft_living": [2000],
                "sqft_lot": [8000],
                "floors": [1.5],
                "sqft_above": [1500],
                "sqft_basement": [500],
                "ppltn_qty": [28000],
                "urbn_ppltn_qty": [22000],
                "sbrbn_ppltn_qty": [6000],
                "farm_ppltn_qty": [0],
                "non_farm_qty": [28000],
                "medn_hshld_incm_amt": [70000],
                "medn_incm_per_prsn_amt": [38000],
                "hous_val_amt": [450000],
                "edctn_less_than_9_qty": [550],
                "edctn_9_12_qty": [2200],
                "edctn_high_schl_qty": [9000],
                "edctn_some_clg_qty": [6500],
                "edctn_assoc_dgre_qty": [3200],
                "edctn_bchlr_dgre_qty": [4500],
                "edctn_prfsnl_qty": [1800],
                "per_urbn": [0.79],
                "per_sbrbn": [0.21],
                "per_farm": [0.0],
                "per_non_farm": [1.0],
                "per_less_than_9": [0.02],
                "per_9_to_12": [0.08],
                "per_hsd": [0.32],
                "per_some_clg": [0.23],
                "per_assoc": [0.11],
                "per_bchlr": [0.16],
                "per_prfsnl": [0.06],
            }
        )

        pred_v1 = model_v1.predict(test_data)[0]
        pred_v2 = model_v2.predict(test_data)[0]

        print(f"\nModel Version Comparison:")
        print(f"Version 1 Prediction: ${pred_v1:,.2f}")
        print(f"Version 2 Prediction: ${pred_v2:,.2f}")
        print(f"Difference: ${abs(pred_v2 - pred_v1):,.2f}")

    except Exception as e:
        print(f"Could not compare versions: {e}")
        print("This is normal if you only have one model version.")


def main():
    """Main demonstration function."""
    print("MLflow Model Usage Demo")
    print("=" * 50)

    # Show experiment information
    print("\n1. Experiment Information:")
    runs = get_experiment_info()

    # Make sample predictions
    print("\n2. Sample Predictions:")
    make_sample_predictions()

    # Compare model versions if available
    print("\n3. Model Version Comparison:")
    compare_model_versions()

    print("\n" + "=" * 50)
    print("Demo completed!")
    print("\nTo view detailed results in MLflow UI:")
    print("1. Run: mlflow ui --backend-store-uri file:./mlruns")
    print("2. Open: http://localhost:5000")
    print("3. Navigate to 'Real Estate Price Prediction' experiment")


if __name__ == "__main__":
    main()
