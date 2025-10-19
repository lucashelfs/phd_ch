"""
Comparison script for V1 vs V2 API endpoints.

This script runs the same test cases through both V1 and V2 APIs to compare
their predictions, performance, and behavior. It provides detailed analysis
of differences between the pickle-based V1 model and MLflow-based V2 model.
"""

import sys
import os
from typing import List, Dict, Any, Tuple

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


def run_parallel_predictions(
    client: APIClient, data_df, sample_size: int = 10, minimal: bool = False
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Run the same predictions through both V1 and V2 APIs.

    Args:
        client: APIClient instance
        data_df: DataFrame with house data
        sample_size: Number of samples to test
        minimal: Use minimal endpoints if True

    Returns:
        Tuple of (v1_results, v2_results)
    """
    endpoint_type = "minimal" if minimal else "full"
    ResponseFormatter.print_subheader(
        f"Running Parallel {endpoint_type.title()} Predictions"
    )

    v1_results = []
    v2_results = []

    print(f"  Testing {sample_size} samples on both V1 and V2 APIs...")

    for i in range(min(sample_size, len(data_df))):
        row = data_df.iloc[i]
        house_data = DataLoader.prepare_house_data(row, minimal=minimal)

        # Test V1
        v1_response = client.predict(house_data, version="v1", minimal=minimal)
        v1_result = v1_response["data"] if v1_response["success"] else None

        # Test V2
        v2_response = client.predict(house_data, version="v2", minimal=minimal)
        v2_result = v2_response["data"] if v2_response["success"] else None

        # Store results
        v1_results.append(v1_result)
        v2_results.append(v2_result)

        # Display progress
        v1_status = "✓" if v1_result else "✗"
        v2_status = "✓" if v2_result else "✗"
        print(f"    Sample {i + 1:2d}: V1 {v1_status} | V2 {v2_status}")

    return v1_results, v2_results


def compare_predictions(
    v1_results: List[Dict[str, Any]],
    v2_results: List[Dict[str, Any]],
    endpoint_type: str = "full",
):
    """
    Compare predictions between V1 and V2 APIs.

    Args:
        v1_results: Results from V1 API
        v2_results: Results from V2 API
        endpoint_type: Type of endpoint tested (full/minimal)
    """
    ResponseFormatter.print_subheader(
        f"Prediction Comparison - {endpoint_type.title()} Endpoints"
    )

    # Filter out failed predictions
    valid_pairs = [(v1, v2) for v1, v2 in zip(v1_results, v2_results) if v1 and v2]

    if not valid_pairs:
        print("  No valid prediction pairs to compare")
        return

    print(f"  Comparing {len(valid_pairs)} successful prediction pairs...")
    print()
    print(
        f"  {'Sample':<8} {'V1 Price':<12} {'V2 Price':<12} {'Difference':<12} {'% Diff':<8} {'V1 Time':<8} {'V2 Time':<8}"
    )
    print("  " + "-" * 80)

    differences = []
    percent_differences = []
    v1_times = []
    v2_times = []

    for i, (v1_result, v2_result) in enumerate(valid_pairs):
        v1_pred = v1_result["prediction"]
        v2_pred = v2_result["prediction"]
        v1_time = v1_result.get("response_time", 0)
        v2_time = v2_result.get("response_time", 0)

        diff = v1_pred - v2_pred
        percent_diff = (diff / v1_pred) * 100 if v1_pred != 0 else 0

        differences.append(abs(diff))
        percent_differences.append(abs(percent_diff))
        v1_times.append(v1_time)
        v2_times.append(v2_time)

        print(
            f"  {i + 1:<8} ${v1_pred:<11,.0f} ${v2_pred:<11,.0f} ${diff:<11,.0f} "
            f"{percent_diff:<7.1f}% {v1_time:<7.3f}s {v2_time:<7.3f}s"
        )

    # Statistical analysis
    if differences:
        import statistics

        print("\n  Statistical Analysis:")
        print(f"    Valid Comparisons: {len(valid_pairs)}")
        print(f"    Average Absolute Difference: ${statistics.mean(differences):,.0f}")
        print(f"    Median Absolute Difference: ${statistics.median(differences):,.0f}")
        print(f"    Max Absolute Difference: ${max(differences):,.0f}")
        print(f"    Min Absolute Difference: ${min(differences):,.0f}")
        if len(differences) > 1:
            print(f"    Std Dev of Differences: ${statistics.stdev(differences):,.0f}")

        print("\n  Percentage Differences:")
        print(
            f"    Average Absolute % Difference: {statistics.mean(percent_differences):.2f}%"
        )
        print(
            f"    Median Absolute % Difference: {statistics.median(percent_differences):.2f}%"
        )
        print(f"    Max Absolute % Difference: {max(percent_differences):.2f}%")

        print("\n  Performance Comparison:")
        print(f"    V1 Average Response Time: {statistics.mean(v1_times):.3f}s")
        print(f"    V2 Average Response Time: {statistics.mean(v2_times):.3f}s")
        time_diff = statistics.mean(v2_times) - statistics.mean(v1_times)
        if time_diff > 0:
            print(f"    V2 is {time_diff:.3f}s slower on average")
        else:
            print(f"    V2 is {abs(time_diff):.3f}s faster on average")


def compare_model_characteristics(
    v1_results: List[Dict[str, Any]], v2_results: List[Dict[str, Any]]
):
    """
    Compare model characteristics and metadata.

    Args:
        v1_results: Results from V1 API
        v2_results: Results from V2 API
    """
    ResponseFormatter.print_subheader("Model Characteristics Comparison")

    # Get first valid result from each API
    v1_sample = next((r for r in v1_results if r), None)
    v2_sample = next((r for r in v2_results if r), None)

    if not v1_sample or not v2_sample:
        print("  Cannot compare - missing valid results from one or both APIs")
        return

    print(f"  {'Characteristic':<25} {'V1 (Pickle)':<20} {'V2 (MLflow)':<20}")
    print("  " + "-" * 65)

    # Compare basic characteristics
    characteristics = [
        ("Model Type", "model_type"),
        ("Model Version", "model_version"),
        ("Features Used", "features_used"),
        ("Prediction Type", "prediction_type"),
    ]

    for char_name, key in characteristics:
        v1_value = v1_sample.get(key, "N/A")
        v2_value = v2_sample.get(key, "N/A")
        print(f"  {char_name:<25} {str(v1_value):<20} {str(v2_value):<20}")

    # Additional V2-specific info
    print("\n  V2 MLflow-Specific Information:")
    if "mlflow_run_id" in v2_sample:
        print(f"    MLflow Run ID: {v2_sample['mlflow_run_id']}")
    if "mlflow_model_uri" in v2_sample:
        print(f"    MLflow Model URI: {v2_sample['mlflow_model_uri']}")


def analyze_agreement_patterns(
    v1_results: List[Dict[str, Any]], v2_results: List[Dict[str, Any]]
):
    """
    Analyze patterns in agreement/disagreement between models.

    Args:
        v1_results: Results from V1 API
        v2_results: Results from V2 API
    """
    ResponseFormatter.print_subheader("Model Agreement Analysis")

    valid_pairs = [(v1, v2) for v1, v2 in zip(v1_results, v2_results) if v1 and v2]

    if len(valid_pairs) < 2:
        print("  Insufficient data for agreement analysis")
        return

    # Categorize differences
    small_diff = 0  # < 5%
    medium_diff = 0  # 5-15%
    large_diff = 0  # > 15%

    for v1_result, v2_result in valid_pairs:
        v1_pred = v1_result["prediction"]
        v2_pred = v2_result["prediction"]

        percent_diff = abs((v1_pred - v2_pred) / v1_pred * 100) if v1_pred != 0 else 0

        if percent_diff < 5:
            small_diff += 1
        elif percent_diff < 15:
            medium_diff += 1
        else:
            large_diff += 1

    total = len(valid_pairs)
    print(f"  Agreement Categories (out of {total} comparisons):")
    print(
        f"    High Agreement (< 5% diff):   {small_diff:2d} ({small_diff / total * 100:.1f}%)"
    )
    print(
        f"    Medium Agreement (5-15% diff): {medium_diff:2d} ({medium_diff / total * 100:.1f}%)"
    )
    print(
        f"    Low Agreement (> 15% diff):   {large_diff:2d} ({large_diff / total * 100:.1f}%)"
    )

    # Overall assessment
    if small_diff / total > 0.8:
        assessment = "Excellent - Models show high agreement"
    elif small_diff / total > 0.6:
        assessment = "Good - Models generally agree"
    elif small_diff / total > 0.4:
        assessment = "Fair - Models show moderate agreement"
    else:
        assessment = "Poor - Models frequently disagree"

    print(f"\n  Overall Assessment: {assessment}")


def generate_summary_report(
    v1_full_results: List[Dict[str, Any]],
    v2_full_results: List[Dict[str, Any]],
    v1_minimal_results: List[Dict[str, Any]],
    v2_minimal_results: List[Dict[str, Any]],
):
    """
    Generate comprehensive summary report.

    Args:
        v1_full_results: V1 full endpoint results
        v2_full_results: V2 full endpoint results
        v1_minimal_results: V1 minimal endpoint results
        v2_minimal_results: V2 minimal endpoint results
    """
    ResponseFormatter.print_subheader("Comprehensive Summary Report")

    # Success rates
    v1_full_success = len([r for r in v1_full_results if r])
    v2_full_success = len([r for r in v2_full_results if r])
    v1_minimal_success = len([r for r in v1_minimal_results if r])
    v2_minimal_success = len([r for r in v2_minimal_results if r])

    total_tests = len(v1_full_results)

    print(f"  Success Rates (out of {total_tests} tests each):")
    print(
        f"    V1 Full Endpoint:     {v1_full_success:2d} ({v1_full_success / total_tests * 100:.1f}%)"
    )
    print(
        f"    V2 Full Endpoint:     {v2_full_success:2d} ({v2_full_success / total_tests * 100:.1f}%)"
    )
    print(
        f"    V1 Minimal Endpoint:  {v1_minimal_success:2d} ({v1_minimal_success / total_tests * 100:.1f}%)"
    )
    print(
        f"    V2 Minimal Endpoint:  {v2_minimal_success:2d} ({v2_minimal_success / total_tests * 100:.1f}%)"
    )

    # Performance summary
    if v1_full_success > 0 and v2_full_success > 0:
        v1_predictions = [r["prediction"] for r in v1_full_results if r]
        v2_predictions = [r["prediction"] for r in v2_full_results if r]

        import statistics

        print("\n  Price Prediction Ranges:")
        print(
            f"    V1 Range: ${min(v1_predictions):,.0f} - ${max(v1_predictions):,.0f}"
        )
        print(
            f"    V2 Range: ${min(v2_predictions):,.0f} - ${max(v2_predictions):,.0f}"
        )
        print(f"    V1 Average: ${statistics.mean(v1_predictions):,.0f}")
        print(f"    V2 Average: ${statistics.mean(v2_predictions):,.0f}")

    # Recommendations
    print("\n  Recommendations:")
    if v1_full_success == total_tests and v2_full_success == total_tests:
        print("    ✓ Both APIs are fully functional")
        print("    ✓ Consider using V2 for MLflow integration benefits")
        print("    ✓ Use V1 for maximum compatibility and stability")
    elif v1_full_success == total_tests:
        print("    ✓ V1 API is fully functional")
        print("    ⚠ V2 API has some issues - check MLflow configuration")
    elif v2_full_success == total_tests:
        print("    ✓ V2 API is fully functional")
        print("    ⚠ V1 API has some issues - check pickle model loading")
    else:
        print("    ⚠ Both APIs have issues - check server configuration")


def main():
    """Main function to run API comparison."""

    ResponseFormatter.print_header("Real Estate API - V1 vs V2 Comparison")

    # Initialize API client
    client = APIClient(DEFAULT_API_URL)

    # Check API availability
    v1_available, v2_available = check_api_availability(client)

    if not v1_available:
        print("V1 API is not available. Cannot run comparison.")
        return

    if not v2_available:
        print("V2 API is not available. Cannot run comparison.")
        print("Please configure V2 API (MLflow champion model) to run comparison.")
        return

    print("\nBoth APIs are available! Running comprehensive comparison...")

    # Get model information for both versions
    v1_info = client.get_model_info("v1")
    v2_info = client.get_model_info("v2")

    if v1_info["success"]:
        ResponseFormatter.print_model_info(v1_info["data"], "v1")
    if v2_info["success"]:
        ResponseFormatter.print_model_info(v2_info["data"], "v2")

    # Load test data
    try:
        print(f"\nLoading test data from {DATA_FILE_PATH}...")
        data_df = DataLoader.load_future_examples(DATA_FILE_PATH)
        print(f"Loaded {len(data_df)} house examples for comparison")

    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # Run parallel tests
    sample_size = min(DEFAULT_SAMPLE_SIZE, len(data_df))
    print(f"\nRunning comparison with {sample_size} samples...")

    # Test full endpoints
    v1_full_results, v2_full_results = run_parallel_predictions(
        client, data_df, sample_size, minimal=False
    )

    # Test minimal endpoints
    v1_minimal_results, v2_minimal_results = run_parallel_predictions(
        client, data_df, sample_size, minimal=True
    )

    # Compare results
    compare_predictions(v1_full_results, v2_full_results, "full")
    compare_predictions(v1_minimal_results, v2_minimal_results, "minimal")

    # Model characteristics comparison
    compare_model_characteristics(v1_full_results, v2_full_results)

    # Agreement analysis
    analyze_agreement_patterns(v1_full_results, v2_full_results)

    # Generate comprehensive summary
    generate_summary_report(
        v1_full_results, v2_full_results, v1_minimal_results, v2_minimal_results
    )

    print("\nAPI comparison completed!")
    print(
        "Both V1 (pickle-based) and V2 (MLflow-based) APIs have been tested and compared."
    )


if __name__ == "__main__":
    main()
