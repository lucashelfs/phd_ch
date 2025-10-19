"""
Test script for V1 API endpoints.

This script demonstrates the V1 API functionality using examples from
data/future_unseen_examples.csv. It tests both full and minimal prediction
endpoints and displays comprehensive results.
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


def test_v1_full_predictions(
    client: APIClient, data_df, sample_size: int = 5
) -> List[Dict[str, Any]]:
    """
    Test V1 full prediction endpoint.

    Args:
        client: APIClient instance
        data_df: DataFrame with house data
        sample_size: Number of samples to test

    Returns:
        List of prediction results
    """
    ResponseFormatter.print_subheader("Testing V1 Full Predictions (/v1/predict)")

    results = []
    predictions = []
    response_times = []

    for i in range(min(sample_size, len(data_df))):
        row = data_df.iloc[i]
        house_data = DataLoader.prepare_house_data(row, minimal=False)

        # Make prediction
        response = client.predict(house_data, version="v1", minimal=False)

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
        ResponseFormatter.print_summary_stats(predictions, response_times, "V1 Full")

    return results


def test_v1_minimal_predictions(
    client: APIClient, data_df, sample_size: int = 5
) -> List[Dict[str, Any]]:
    """
    Test V1 minimal prediction endpoint.

    Args:
        client: APIClient instance
        data_df: DataFrame with house data
        sample_size: Number of samples to test

    Returns:
        List of prediction results
    """
    ResponseFormatter.print_subheader(
        "Testing V1 Minimal Predictions (/v1/predict/minimal)"
    )

    results = []
    predictions = []
    response_times = []

    for i in range(min(sample_size, len(data_df))):
        row = data_df.iloc[i]
        house_data = DataLoader.prepare_house_data(row, minimal=True)

        # Make prediction
        response = client.predict(house_data, version="v1", minimal=True)

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
        ResponseFormatter.print_summary_stats(predictions, response_times, "V1 Minimal")

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
    ResponseFormatter.print_subheader("Comparing Full vs Minimal Predictions")

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


def main():
    """Main function to run V1 API tests."""

    ResponseFormatter.print_header("Real Estate API - V1 Endpoint Testing")

    # Initialize API client
    client = APIClient(DEFAULT_API_URL)

    # Check API availability
    v1_available, v2_available = check_api_availability(client)

    if not v1_available:
        print("V1 API is not available. Please ensure the API server is running.")
        return

    # Get V1 model information
    model_info_response = client.get_model_info("v1")
    if model_info_response["success"]:
        ResponseFormatter.print_model_info(model_info_response["data"], "v1")
    else:
        print(f"Could not get V1 model info: {model_info_response['error']}")

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

        # Debug: Show formatted data for first sample
        print("\nDebug - Formatted data for first sample:")
        formatted_full = DataLoader.prepare_house_data(sample_row, minimal=False)
        formatted_minimal = DataLoader.prepare_house_data(sample_row, minimal=True)
        print(f"  Full format: {formatted_full}")
        print(f"  Minimal format: {formatted_minimal}")

    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print(
            "Please ensure the data file exists and the API server is running from the correct directory."
        )
        return

    # Test V1 endpoints
    print(f"\nTesting V1 API with {DEFAULT_SAMPLE_SIZE} samples...")

    # Test full predictions
    full_results = test_v1_full_predictions(client, data_df, DEFAULT_SAMPLE_SIZE)

    # Test minimal predictions
    minimal_results = test_v1_minimal_predictions(client, data_df, DEFAULT_SAMPLE_SIZE)

    # Compare results
    compare_full_vs_minimal(full_results, minimal_results)

    # Final summary
    ResponseFormatter.print_subheader("V1 API Test Summary")
    print(f"  Total samples tested: {DEFAULT_SAMPLE_SIZE}")
    print(f"  Full predictions successful: {len(full_results)}")
    print(f"  Minimal predictions successful: {len(minimal_results)}")

    if full_results and minimal_results:
        print("  V1 API is working correctly with both endpoints")
        print(f"  Model type: {full_results[0].get('model_type', 'Unknown')}")
        print(f"  Features used (full): {full_results[0].get('features_used', 0)}")
        print(
            f"  Features used (minimal): {minimal_results[0].get('features_used', 0)}"
        )
    else:
        print("  Some endpoints may have issues - check API server logs")

    print("\nV1 API testing completed!")


if __name__ == "__main__":
    main()
