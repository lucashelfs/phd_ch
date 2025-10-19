# API Testing Examples

This directory contains comprehensive test scripts for the Real Estate Price Prediction API. These scripts demonstrate how to use both V1 (pickle-based) and V2 (MLflow-based) endpoints with real data examples.

## Overview

The testing suite consists of four main components:

1. **`api_test_utils.py`** - Shared utilities and helper functions
2. **`test_v1_api.py`** - Tests V1 API endpoints (pickle-based models)
3. **`test_v2_api.py`** - Tests V2 API endpoints (MLflow-based models)
4. **`compare_api_versions.py`** - Compares V1 vs V2 API performance and predictions

## Prerequisites

### API Server
Ensure the API server is running:
```bash
# From the repo root directory
docker-compose up -d
```

### Data File
The scripts use `data/future_unseen_examples.csv` which contains 100 house examples with all required features (except price, which is what we're predicting).

### Python Dependencies
The scripts require:
- `pandas` - Data loading and manipulation
- `requests` - HTTP client for API calls
- Standard library modules: `json`, `time`, `statistics`, `typing`

## Usage

### 1. Test V1 API Only

Test the V1 (pickle-based) API endpoints:

```bash
cd repo/examples
python test_v1_api.py
```

**What it does:**
- Tests `/v1/info` endpoint for model information
- Tests `/v1/predict` with full house features (10 samples)
- Tests `/v1/predict/minimal` with core features only (10 samples)
- Compares full vs minimal predictions
- Shows performance statistics and response times

**Sample Output:**
```
================================================================================
                    Real Estate API - V1 Endpoint Testing
================================================================================
API Health Check: OK
Available API Versions: v1

V1 Model Information:
  Model Type: KNeighborsRegressor
  Model Version: 1.0.0
  Total Features: 33
  Demographics Zipcodes: 70

Testing V1 Full Predictions (/v1/predict)
------------------------------------------------------------
  Sample  1: 4BR/1.0BA, 1,680sqft, Zipcode 98118 → $425,000 (KNeighborsRegressor, 33 features, 0.045s)
  Sample  2: 3BR/2.5BA, 2,220sqft, Zipcode 98115 → $650,000 (KNeighborsRegressor, 33 features, 0.032s)
  ...
```

### 2. Test V2 API Only

Test the V2 (MLflow-based) API endpoints:

```bash
cd repo/examples
python test_v2_api.py
```

**What it does:**
- Checks if V2 API is available (requires MLflow champion model)
- Tests `/v2/info` endpoint for MLflow model information
- Tests `/v2/predict` and `/v2/predict/minimal` endpoints
- Shows MLflow-specific metadata (run ID, model URI, etc.)

**Note:** V2 API requires:
1. MLflow server running
2. Champion model registered in MLflow
3. `CHAMPION_MODEL_MLFLOW_URI` configured in `.env`

### 3. Compare V1 vs V2 APIs

Run comprehensive comparison between both API versions:

```bash
cd repo/examples
python compare_api_versions.py
```

**What it does:**
- Tests the same house examples on both V1 and V2 APIs
- Compares predictions side-by-side
- Analyzes prediction differences and agreement patterns
- Compares response times and performance
- Generates comprehensive summary report

**Sample Output:**
```
================================================================================
                    Real Estate API - V1 vs V2 Comparison
================================================================================

Running Parallel Full Predictions
------------------------------------------------------------
  Testing 10 samples on both V1 and V2 APIs...
    Sample  1: V1 ✓ | V2 ✓
    Sample  2: V1 ✓ | V2 ✓
    ...

Prediction Comparison - Full Endpoints
------------------------------------------------------------
  Sample   V1 Price     V2 Price     Difference   % Diff   V1 Time  V2 Time
  --------------------------------------------------------------------------------
  1        $425,000     $438,500     $13,500      3.2%     0.045s   0.052s
  2        $650,000     $642,000     -$8,000      -1.2%    0.032s   0.048s
  ...
```

## Configuration

### Sample Size
By default, scripts test 10 samples. You can modify this in `api_test_utils.py`:

```python
DEFAULT_SAMPLE_SIZE = 10  # Change this value
```

### API URL
By default, scripts connect to `http://localhost:8000`. You can modify this in `api_test_utils.py`:

```python
DEFAULT_API_URL = "http://localhost:8000"  # Change this value
```

### Timeout Settings
Request timeout is set to 30 seconds by default. Modify in `api_test_utils.py`:

```python
DEFAULT_TIMEOUT = 30  # Change this value
```

## Understanding the Output

### Prediction Format
Each prediction shows:
- **Sample number**: Sequential test number
- **House details**: Bedrooms/Bathrooms, square footage, zipcode
- **Predicted price**: Model's price prediction
- **Model info**: Model type, features used, response time

Example: `Sample  1: 4BR/1.0BA, 1,680sqft, Zipcode 98118 → $425,000 (KNeighborsRegressor, 33 features, 0.045s)`

### Comparison Metrics
- **Absolute Difference**: Dollar difference between V1 and V2 predictions
- **Percentage Difference**: Relative difference as percentage
- **Agreement Categories**:
  - High Agreement: < 5% difference
  - Medium Agreement: 5-15% difference
  - Low Agreement: > 15% difference

### Performance Metrics
- **Response Time**: Time taken for each API call
- **Success Rate**: Percentage of successful predictions
- **Price Range**: Min/max predicted prices

## Troubleshooting

### Common Issues

**"V1 API is not available"**
- Ensure API server is running: `docker-compose ps`
- Check API health: `curl http://localhost:8000/health`

**"V2 API is not available"**
- Check if MLflow champion model is configured
- Verify `CHAMPION_MODEL_MLFLOW_URI` in `.env` file
- Ensure MLflow server is running: `curl http://localhost:5000`

**"Could not load data"**
- Ensure you're running from the correct directory
- Check that `data/future_unseen_examples.csv` exists
- Verify file permissions

**Connection errors**
- Check if API server is running on the expected port
- Verify firewall settings
- Try increasing timeout values

### Debug Mode

For more detailed error information, you can modify the scripts to show full error details:

```python
# In any of the test scripts, add this for debugging:
import traceback

try:
    # existing code
except Exception as e:
    print(f"Detailed error: {traceback.format_exc()}")
```

## Extending the Scripts

### Adding New Test Cases
To test with different data:

1. Create a new CSV file with the same format as `future_unseen_examples.csv`
2. Modify `DATA_FILE_PATH` in the scripts
3. Run the tests

### Adding New Metrics
To add custom analysis:

1. Modify the comparison functions in `compare_api_versions.py`
2. Add new statistical calculations
3. Update the summary report generation

### Testing Different Endpoints
To test additional endpoints:

1. Add new methods to the `APIClient` class in `api_test_utils.py`
2. Create test functions in the individual test scripts
3. Update the comparison logic as needed

## Integration with CI/CD

These scripts can be integrated into automated testing pipelines:

```bash
# Example CI/CD usage
python test_v1_api.py > v1_test_results.log
python test_v2_api.py > v2_test_results.log
python compare_api_versions.py > comparison_results.log
```

The scripts return appropriate exit codes for automated testing environments.
