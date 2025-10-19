"""
Test script for V2 API endpoints.

This script demonstrates the V2 API functionality using examples from
data/future_unseen_examples.csv. It tests both full and minimal prediction
endpoints using MLflow-based models and displays comprehensive results.
"""

import sys
import os
from typing import List, Dict, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from api_test_utils import (
    APIClient,
    DataLoader,
    ResponseFormatter,
    check_api_availability,
    DEFAULT_API_URL,
    DEFAULT_SAMPLE_SIZE,
    DATA_FILE_PATH,
)


def test_v2_full_predictions(
    client: APIClient, data_df, sample_size: int = 5
) -> List[Dict[str, Any]]:
    """
    Test V2 full prediction endpoint.

    Args:
        client: APIClient instance
        data_df: DataFrame with house data
        sample_size: Number of samples to test

    Returns:
        List of prediction results
    """
    ResponseFormatter.print_subheader("Testing V2 Full Predictions (/v2/predict)")

    results = []
    predictions = []
    response_times = []

    for i in range(min(sample_size, len(data_df))):
        row = data_df.iloc[i]
        house_data = DataLoader.prepare_house_data(row, minimal=False)

        # Make prediction
        response = client.predict(house_data, version="v2", minimal=False)

        if response["success"]:
            result = response["data"]
            results.append(result)
            predictions.append(result["prediction"])
            response_times.append(result["response_time"])

            # Display result
            formatted_result = ResponseFormatter.format_prediction_result(
                result, house_data, i
            )
            print(f"  {formatted_result}")
        else:
            print(f"  Sample {i + 1:2d}: FAILED - {response['error']}")

    # Print summary
    if predictions:
        ResponseFormatter.print_summary_stats(predictions, response_times, "V2 Full")

    return results


def test_v2_minimal_predictions(
    client: APIClient, data_df, sample_size: int = 5
) -> List[Dict[str, Any]]:
    """
    Test V2 minimal prediction endpoint.

    Args:
        client: APIClient instance
        data_df: DataFrame with house data
        sample_size: Number of samples to test

    Returns:
        List of prediction results
    """
    ResponseFormatter.print_subheader(
        "Testing V2 Minimal Predictions (/v2/predict/minimal)"
    )

    results = []
    predictions = []
    response_times = []

    for i in range(min(sample_size, len(data_df))):
        row = data_df.iloc[i]
        house_data = DataLoader.prepare_house_data(row, minimal=True)

        # Make prediction
        response = client.predict(house_data, version="v2", minimal=True)

        if response["success"]:
            result = response["data"]
            results.append(result)
            predictions.append(result["prediction"])
            response_times.append(result["response_time"])

            # Display result
            formatted_result = ResponseFormatter.format_prediction_result(
                result, house_data, i
            )
            print(f"  {formatted_result}")
        else:
            print(f"  Sample {i + 1:2d}: FAILED - {response['error']}")

    # Print summary
    if predictions:
        ResponseFormatter.print_summary_stats(predictions, response_times, "V2 Minimal")

    return results


def compare_full_vs_minimal(
    full_results: List[Dict[str, Any]], minimal_results: List[Dict[str, Any]]
):
    """
    Compare full vs minimal prediction results.

    Args:
        full_results: Results from full predictions
        minimal_results: Results from minimal predictions
    """
    ResponseFormatter.print_subheader("Comparing V2 Full vs Minimal Predictions")

    if not full_results or not minimal_results:
        print("  Cannot compare - insufficient data from both endpoints")
        return

    # Compare predictions for matching samples
    min_samples = min(len(full_results), len(minimal_results))

    print(
        f"  {'Sample':<8} {'Full Price':<12} {'Minimal Price':<14} {'Difference':<12} {'% Diff':<8}"
    )
    print("  " + "-" * 60)

    differences = []
    percent_differences = []

    for i in range(min_samples):
        full_pred = full_results[i]["prediction"]
        minimal_pred = minimal_results[i]["prediction"]

        diff = full_pred - minimal_pred
        percent_diff = (diff / full_pred) * 100 if full_pred != 0 else 0

        differences.append(abs(diff))
        percent_differences.append(abs(percent_diff))

        print(
            f"  {i + 1:<8} ${full_pred:<11,.0f} ${minimal_pred:<13,.0f} ${diff:<11,.0f} {percent_diff:<7.1f}%"
        )

    # Summary statistics
    if differences:
        import statistics

        print("\n  Comparison Summary:")
        print(f"    Average Absolute Difference: ${statistics.mean(differences):,.0f}")
        print(f"    Max Absolute Difference: ${max(differences):,.0f}")
        print(
            f"    Average Absolute % Difference: {statistics.mean(percent_differences):.1f}%"
        )
        print(f"    Max Absolute % Difference: {max(percent_differences):.1f}%")

        # Feature usage comparison
        full_features = full_results[0].get("features_used", 0)
        minimal_features = minimal_results[0].get("features_used", 0)
        print(f"    Full Endpoint Features: {full_features}")
        print(f"    Minimal Endpoint Features: {minimal_features}")


def display_mlflow_info(model_info: Dict[str, Any]):
    """
    Display MLflow-specific model information.

    Args:
        model_info: Model information from V2 API
    """
    ResponseFormatter.print_subheader("MLflow Model Details")

    # Standard model info
    print(f"  Model Type: {model_info.get('model_type', 'Unknown')}")
    print(f"  Model Version: {model_info.get('model_version', 'Unknown')}")
    print(f"  Total Features: {model_info.get('total_features', 0)}")
    print(f"  Demographics Zipcodes: {model_info.get('demographics_zipcodes', 0)}")

    # MLflow-specific info (if available)
    if "mlflow_run_id" in model_info:
        print(f"  MLflow Run ID: {model_info['mlflow_run_id']}")
    if "mlflow_model_uri" in model_info:
        print(f"  MLflow Model URI: {model_info['mlflow_model_uri']}")
    if "champion_metrics" in model_info:
        metrics = model_info["champion_metrics"]
        print("  Champion Model Metrics:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"    {metric}: {value:.4f}")
            else:
                print(f"    {metric}: {value}")


def main():
    """Main function to run V2 API tests."""

    ResponseFormatter.print_header("Real Estate API - V2 Endpoint Testing")

    # Initialize API client
    client = APIClient(DEFAULT_API_URL)

    # Check API availability
    v1_available, v2_available = check_api_availability(client)

    if not v2_available:
        print("\nV2 API is not available.")
        print("This could be because:")
        print("  1. CHAMPION_MODEL_MLFLOW_URI is not configured in .env")
        print("  2. MLflow server is not running")
        print("  3. No champion model has been registered in MLflow")
        print("\nTo enable V2 API:")
        print("  1. Train and register a model in MLflow")
        print("  2. Set CHAMPION_MODEL_MLFLOW_URI in .env file")
        print("  3. Restart the API server")
        return

    print("\nV2 API is available! Testing MLflow-based predictions...")

    # Get V2 model information
    model_info_response = client.get_model_info("v2")
    if model_info_response["success"]:
        model_info = model_info_response["data"]
        ResponseFormatter.print_model_info(model_info, "v2")
        display_mlflow_info(model_info)
    else:
        print(f"Could not get V2 model info: {model_info_response['error']}")
        return

    # Load test data
    try:
        print(f"\nLoading test data from {DATA_FILE_PATH}...")
        data_df = DataLoader.load_future_examples(DATA_FILE_PATH)
        print(f"Loaded {len(data_df)} house examples")

        # Show sample of data
        print("\nSample house features:")
        sample_row = data_df.iloc[0]
        print(
            f"  Bedrooms: {sample_row['bedrooms']}, Bathrooms: {sample_row['bathrooms']}"
        )
        print(f"  Living Space: {sample_row['sqft_living']:,} sqft")
        print(f"  Lot Size: {sample_row['sqft_lot']:,} sqft")
        print(f"  Year Built: {sample_row['yr_built']}")
        print(f"  Zipcode: {sample_row['zipcode']}")

    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print(
            "Please ensure the data file exists and the API server is running from the correct directory."
        )
        return

    # Test V2 endpoints
    print(f"\nTesting V2 API with {DEFAULT_SAMPLE_SIZE} samples...")

    # Test full predictions
    full_results = test_v2_full_predictions(client, data_df, DEFAULT_SAMPLE_SIZE)

    # Test minimal predictions
    minimal_results = test_v2_minimal_predictions(client, data_df, DEFAULT_SAMPLE_SIZE)

    # Compare results
    compare_full_vs_minimal(full_results, minimal_results)

    # Final summary
    ResponseFormatter.print_subheader("V2 API Test Summary")
    print(f"  Total samples tested: {DEFAULT_SAMPLE_SIZE}")
    print(f"  Full predictions successful: {len(full_results)}")
    print(f"  Minimal predictions successful: {len(minimal_results)}")

    if full_results and minimal_results:
        print("  V2 API is working correctly with both endpoints")
        print(f"  Model type: {full_results[0].get('model_type', 'Unknown')}")
        print(f"  Features used (full): {full_results[0].get('features_used', 0)}")
        print(
            f"  Features used (minimal): {minimal_results[0].get('features_used', 0)}"
        )

        # Show MLflow-specific metadata from predictions
        if "model_version" in full_results[0]:
            print(f"  MLflow Model Version: {full_results[0]['model_version']}")

    else:
        print("  Some endpoints may have issues - check API server logs")

    print("\nV2 API testing completed!")
    print("MLflow champion model is working correctly!")


if __name__ == "__main__":
    main()
