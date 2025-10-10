"""
API validation script for the Real Estate Price Prediction API.

This script tests the API endpoints using examples from the
future_unseen_examples.csv file to validate functionality.
"""

import json
import requests
import pandas as pd
from typing import Dict, Any, List
import time
import sys
from pathlib import Path


class APITester:
    """Test suite for the Real Estate Price Prediction API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the API tester with base URL."""
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.test_results = []
    
    def log_result(self, test_name: str, success: bool, message: str, details: Dict[str, Any] = None):
        """Log test result."""
        result = {
            "test": test_name,
            "success": success,
            "message": message,
            "details": details or {},
            "timestamp": time.time()
        }
        self.test_results.append(result)
        
        status = "PASS" if success else "FAIL"
        print(f"[{status}] {test_name}: {message}")
        if details and not success:
            print(f"   Details: {details}")
    
    def test_health_check(self) -> bool:
        """Test the health check endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    self.log_result("Health Check", True, "API is healthy")
                    return True
                else:
                    self.log_result("Health Check", False, f"API status: {data.get('status')}", data)
                    return False
            else:
                self.log_result("Health Check", False, f"HTTP {response.status_code}", {"response": response.text})
                return False
                
        except Exception as e:
            self.log_result("Health Check", False, f"Connection failed: {str(e)}")
            return False
    
    def test_root_endpoint(self) -> bool:
        """Test the root endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if "version" in data and "endpoints" in data:
                    self.log_result("Root Endpoint", True, f"API version {data['version']}")
                    return True
                else:
                    self.log_result("Root Endpoint", False, "Missing expected fields", data)
                    return False
            else:
                self.log_result("Root Endpoint", False, f"HTTP {response.status_code}", {"response": response.text})
                return False
                
        except Exception as e:
            self.log_result("Root Endpoint", False, f"Request failed: {str(e)}")
            return False
    
    def test_model_info(self) -> bool:
        """Test the model info endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/v1/info", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ["model_type", "model_version", "total_features", "features"]
                
                if all(field in data for field in required_fields):
                    self.log_result("Model Info", True, 
                                  f"{data['model_type']} with {data['total_features']} features")
                    return True
                else:
                    missing = [f for f in required_fields if f not in data]
                    self.log_result("Model Info", False, f"Missing fields: {missing}", data)
                    return False
            else:
                self.log_result("Model Info", False, f"HTTP {response.status_code}", {"response": response.text})
                return False
                
        except Exception as e:
            self.log_result("Model Info", False, f"Request failed: {str(e)}")
            return False
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        """Load test data from future_unseen_examples.csv."""
        try:
            data_path = Path("data/future_unseen_examples.csv")
            if not data_path.exists():
                self.log_result("Load Test Data", False, f"Test data file not found: {data_path}")
                return []
            
            df = pd.read_csv(data_path)
            
            # Convert to list of dictionaries and take first 5 examples
            test_examples = df.head(5).to_dict('records')
            
            # Convert numpy types to Python types for JSON serialization
            for example in test_examples:
                for key, value in example.items():
                    if pd.isna(value):
                        example[key] = None
                    elif key == 'zipcode':
                        # Convert zipcode to string as expected by API
                        example[key] = str(int(value)) if pd.notna(value) else None
                    elif isinstance(value, (pd.Int64Dtype, pd.Float64Dtype)):
                        example[key] = float(value) if pd.notna(value) else None
                    else:
                        example[key] = value
            
            self.log_result("Load Test Data", True, f"Loaded {len(test_examples)} test examples")
            return test_examples
            
        except Exception as e:
            self.log_result("Load Test Data", False, f"Failed to load test data: {str(e)}")
            return []
    
    def test_prediction_endpoint(self, test_data: List[Dict[str, Any]]) -> bool:
        """Test the main prediction endpoint."""
        if not test_data:
            self.log_result("Prediction Endpoint", False, "No test data available")
            return False
        
        success_count = 0
        total_tests = min(3, len(test_data))  # Test first 3 examples
        
        for i, example in enumerate(test_data[:total_tests]):
            try:
                response = self.session.post(
                    f"{self.base_url}/v1/predict",
                    json=example,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    required_fields = ["prediction", "model_version", "model_type", "zipcode"]
                    
                    if all(field in data for field in required_fields):
                        prediction = data["prediction"]
                        zipcode = data["zipcode"]
                        self.log_result(f"Prediction {i+1}", True, 
                                      f"Predicted ${prediction:,.2f} for zipcode {zipcode}")
                        success_count += 1
                    else:
                        missing = [f for f in required_fields if f not in data]
                        self.log_result(f"Prediction {i+1}", False, f"Missing fields: {missing}", data)
                else:
                    error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
                    self.log_result(f"Prediction {i+1}", False, f"HTTP {response.status_code}", {"response": error_data})
                    
            except Exception as e:
                self.log_result(f"Prediction {i+1}", False, f"Request failed: {str(e)}")
        
        overall_success = success_count == total_tests
        self.log_result("Prediction Endpoint", overall_success, 
                       f"{success_count}/{total_tests} predictions successful")
        return overall_success
    
    def test_minimal_prediction_endpoint(self, test_data: List[Dict[str, Any]]) -> bool:
        """Test the minimal prediction endpoint."""
        if not test_data:
            self.log_result("Minimal Prediction Endpoint", False, "No test data available")
            return False
        
        # Core features for minimal endpoint
        core_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
                        'sqft_above', 'sqft_basement', 'zipcode']
        
        success_count = 0
        total_tests = min(2, len(test_data))  # Test first 2 examples
        
        for i, example in enumerate(test_data[:total_tests]):
            try:
                # Filter to only core features
                minimal_example = {k: v for k, v in example.items() if k in core_features}
                
                response = self.session.post(
                    f"{self.base_url}/v1/predict/minimal",
                    json=minimal_example,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    required_fields = ["prediction", "prediction_type", "core_features_used"]
                    
                    if all(field in data for field in required_fields):
                        prediction = data["prediction"]
                        features_used = data["core_features_used"]
                        self.log_result(f"Minimal Prediction {i+1}", True, 
                                      f"Predicted ${prediction:,.2f} using {features_used} core features")
                        success_count += 1
                    else:
                        missing = [f for f in required_fields if f not in data]
                        self.log_result(f"Minimal Prediction {i+1}", False, f"Missing fields: {missing}", data)
                else:
                    error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
                    self.log_result(f"Minimal Prediction {i+1}", False, f"HTTP {response.status_code}", {"response": error_data})
                    
            except Exception as e:
                self.log_result(f"Minimal Prediction {i+1}", False, f"Request failed: {str(e)}")
        
        overall_success = success_count == total_tests
        self.log_result("Minimal Prediction Endpoint", overall_success, 
                       f"{success_count}/{total_tests} minimal predictions successful")
        return overall_success
    
    def test_error_handling(self) -> bool:
        """Test error handling with invalid data."""
        try:
            # Test with invalid zipcode
            invalid_data = {
                "bedrooms": 4,
                "bathrooms": 2.5,
                "sqft_living": 2630,
                "sqft_lot": 4501,
                "floors": 2.0,
                "sqft_above": 2630,
                "sqft_basement": 0,
                "yr_built": 2015,
                "yr_renovated": 0,
                "zipcode": "invalid",  # Invalid zipcode
                "lat": 47.7748,
                "long": -122.244
            }
            
            response = self.session.post(
                f"{self.base_url}/v1/predict",
                json=invalid_data,
                timeout=10
            )
            
            if response.status_code == 422:  # Validation error expected
                self.log_result("Error Handling", True, "Correctly rejected invalid zipcode")
                return True
            else:
                self.log_result("Error Handling", False, 
                               f"Expected 422, got {response.status_code}", {"response": response.text})
                return False
                
        except Exception as e:
            self.log_result("Error Handling", False, f"Request failed: {str(e)}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all API tests."""
        print(f"Starting API tests for {self.base_url}")
        print("=" * 60)
        
        # Basic connectivity tests
        if not self.test_health_check():
            print("\nHealth check failed - API may not be running")
            return False
        
        self.test_root_endpoint()
        self.test_model_info()
        
        # Load test data
        test_data = self.load_test_data()
        
        # Prediction tests
        self.test_prediction_endpoint(test_data)
        self.test_minimal_prediction_endpoint(test_data)
        
        # Error handling test
        self.test_error_handling()
        
        # Summary
        print("\n" + "=" * 60)
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        
        print(f"Test Summary: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("All tests passed! API is working correctly.")
            return True
        else:
            print("Some tests failed. Check the output above for details.")
            return False
    
    def save_results(self, filename: str = "test_results.json"):
        """Save test results to a JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump(self.test_results, f, indent=2, default=str)
            print(f"Test results saved to {filename}")
        except Exception as e:
            print(f"Failed to save test results: {e}")


def main():
    """Main function to run API tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the Real Estate Price Prediction API")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="Base URL of the API (default: http://localhost:8000)")
    parser.add_argument("--save-results", action="store_true",
                       help="Save test results to JSON file")
    
    args = parser.parse_args()
    
    tester = APITester(args.url)
    success = tester.run_all_tests()
    
    if args.save_results:
        tester.save_results()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
