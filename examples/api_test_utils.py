"""
Shared utilities for API testing scripts.

This module provides common functionality for testing the Real Estate API
endpoints, including data loading, API client, and response formatting.
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)


class APIClient:
    """Client for interacting with the Real Estate API."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        """
        Initialize API client.

        Args:
            base_url: Base URL for the API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # Configure session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def check_health(self) -> Dict[str, Any]:
        """Check API health status."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_versions(self) -> Dict[str, Any]:
        """Get available API versions."""
        try:
            response = self.session.get(
                f"{self.base_url}/versions", timeout=self.timeout
            )
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_model_info(self, version: str = "v1") -> Dict[str, Any]:
        """Get model information for specified version."""
        try:
            response = self.session.get(
                f"{self.base_url}/{version}/info", timeout=self.timeout
            )
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def predict(
        self, house_data: Dict[str, Any], version: str = "v1", minimal: bool = False
    ) -> Dict[str, Any]:
        """
        Make prediction request.

        Args:
            house_data: House features dictionary
            version: API version (v1 or v2)
            minimal: Use minimal endpoint if True

        Returns:
            Dictionary with success status and response data or error
        """
        endpoint = f"/{version}/predict"
        if minimal:
            endpoint += "/minimal"

        try:
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}{endpoint}",
                json=house_data,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
            response_time = time.time() - start_time

            # Handle HTTP errors with detailed error information
            if not response.ok:
                try:
                    error_detail = response.json()
                    if isinstance(error_detail, dict) and "detail" in error_detail:
                        error_msg = (
                            f"HTTP {response.status_code}: {error_detail['detail']}"
                        )
                    else:
                        error_msg = f"HTTP {response.status_code}: {error_detail}"
                except (ValueError, KeyError, TypeError):
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                return {"success": False, "error": error_msg}

            result = response.json()
            result["response_time"] = response_time

            return {"success": True, "data": result}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Request failed: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {str(e)}"}


class DataLoader:
    """Utility for loading and processing test data."""

    @staticmethod
    def load_future_examples(
        file_path: str, sample_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load future examples from CSV file.

        Args:
            file_path: Path to the CSV file
            sample_size: Number of samples to load (None for all)

        Returns:
            DataFrame with house features
        """
        try:
            df = pd.read_csv(file_path)

            if sample_size is not None:
                df = df.head(sample_size)

            return df
        except Exception as e:
            raise FileNotFoundError(f"Could not load data from {file_path}: {e}")

    @staticmethod
    def prepare_house_data(row: pd.Series, minimal: bool = False) -> Dict[str, Any]:
        """
        Convert DataFrame row to API request format.

        Args:
            row: Pandas Series with house features
            minimal: If True, include only minimal features

        Returns:
            Dictionary formatted for API request
        """
        if minimal:
            # Core features for minimal endpoint
            return {
                "bedrooms": int(float(row["bedrooms"])),
                "bathrooms": float(row["bathrooms"]),
                "sqft_living": int(float(row["sqft_living"])),
                "sqft_lot": int(float(row["sqft_lot"])),
                "floors": float(row["floors"]),
                "sqft_above": int(float(row["sqft_above"])),
                "sqft_basement": int(float(row["sqft_basement"])),
                "zipcode": str(int(float(row["zipcode"]))).zfill(5),
            }
        else:
            # Full features for complete endpoint
            return {
                "bedrooms": int(float(row["bedrooms"])),
                "bathrooms": float(row["bathrooms"]),
                "sqft_living": int(float(row["sqft_living"])),
                "sqft_lot": int(float(row["sqft_lot"])),
                "floors": float(row["floors"]),
                "waterfront": int(float(row["waterfront"])),
                "view": int(float(row["view"])),
                "condition": int(float(row["condition"])),
                "grade": int(float(row["grade"])),
                "sqft_above": int(float(row["sqft_above"])),
                "sqft_basement": int(float(row["sqft_basement"])),
                "yr_built": int(float(row["yr_built"])),
                "yr_renovated": int(float(row["yr_renovated"])),
                "zipcode": str(int(float(row["zipcode"]))).zfill(5),
                "lat": float(row["lat"]),
                "long": float(row["long"]),
                "sqft_living15": int(float(row["sqft_living15"])),
                "sqft_lot15": int(float(row["sqft_lot15"])),
            }


class ResponseFormatter:
    """Utility for formatting API responses and results."""

    @staticmethod
    def print_header(title: str, width: int = 80):
        """Print formatted header."""
        print("\n" + "=" * width)
        print(f"{title:^{width}}")
        print("=" * width)

    @staticmethod
    def print_subheader(title: str, width: int = 60):
        """Print formatted subheader."""
        print(f"\n{title}")
        print("-" * width)

    @staticmethod
    def format_prediction_result(
        result: Dict[str, Any], house_data: Dict[str, Any], index: int
    ) -> str:
        """
        Format prediction result for display.

        Args:
            result: API response data
            house_data: Original house data sent to API
            index: Sample index

        Returns:
            Formatted string for display
        """
        if not result:
            return f"Sample {index + 1}: Failed to get prediction"

        prediction = result.get("prediction", 0)
        model_type = result.get("model_type", "Unknown")
        features_used = result.get("features_used", 0)
        zipcode = result.get("zipcode", house_data.get("zipcode", "Unknown"))
        response_time = result.get("response_time", 0)

        # Format house details
        bedrooms = house_data.get("bedrooms", 0)
        bathrooms = house_data.get("bathrooms", 0)
        sqft_living = house_data.get("sqft_living", 0)

        return (
            f"Sample {index + 1:2d}: {bedrooms}BR/{bathrooms}BA, {sqft_living:,}sqft, "
            f"Zipcode {zipcode} → ${prediction:,.0f} "
            f"({model_type}, {features_used} features, {response_time:.3f}s)"
        )

    @staticmethod
    def print_model_info(info: Dict[str, Any], version: str):
        """Print formatted model information."""
        print(f"\n{version.upper()} Model Information:")
        print(f"  Model Type: {info.get('model_type', 'Unknown')}")
        print(f"  Model Version: {info.get('model_version', 'Unknown')}")
        print(f"  Total Features: {info.get('total_features', 0)}")
        print(f"  Demographics Zipcodes: {info.get('demographics_zipcodes', 0)}")

        # Show first few features
        features = info.get("features", [])
        if features:
            print(
                f"  Sample Features: {', '.join(features[:5])}{'...' if len(features) > 5 else ''}"
            )

    @staticmethod
    def print_summary_stats(
        predictions: List[float], response_times: List[float], version: str
    ):
        """Print summary statistics."""
        if not predictions:
            print(f"\nNo successful predictions for {version}")
            return

        import statistics

        print(f"\n{version.upper()} Summary Statistics:")
        print(f"  Successful Predictions: {len(predictions)}")
        print(f"  Price Range: ${min(predictions):,.0f} - ${max(predictions):,.0f}")
        print(f"  Average Price: ${statistics.mean(predictions):,.0f}")
        print(f"  Median Price: ${statistics.median(predictions):,.0f}")
        if len(predictions) > 1:
            print(f"  Price Std Dev: ${statistics.stdev(predictions):,.0f}")

        if response_times:
            print(f"  Average Response Time: {statistics.mean(response_times):.3f}s")
            print(f"  Total Response Time: {sum(response_times):.3f}s")


def check_api_availability(client: APIClient) -> Tuple[bool, bool]:
    """
    Check which API versions are available.

    Args:
        client: APIClient instance

    Returns:
        Tuple of (v1_available, v2_available)
    """
    # Check health first
    health = client.check_health()
    if not health["success"]:
        print(f"API Health Check Failed: {health['error']}")
        return False, False

    print("API Health Check: OK")

    # Check available versions
    versions = client.get_versions()
    if not versions["success"]:
        print(f"Could not get version info: {versions['error']}")
        return False, False

    available_versions = versions["data"].get("available_versions", [])
    v1_available = "v1" in available_versions
    v2_available = "v2" in available_versions

    print(f"Available API Versions: {', '.join(available_versions)}")

    return v1_available, v2_available


# Configuration constants
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 30
DEFAULT_SAMPLE_SIZE = 10
DATA_FILE_PATH = os.path.join(_REPO_ROOT, "data", "future_unseen_examples.csv")
