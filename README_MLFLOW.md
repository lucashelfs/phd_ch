# MLflow Enhanced Model Training

This document describes the MLflow-enhanced version of the real estate price prediction model training script.

## Overview

`create_model_mlflow.py` is an enhanced version of the original `create_model.py` script that includes comprehensive MLflow experiment tracking capabilities. It provides the same functionality as the original script while adding enterprise-grade experiment management.

## Features

### MLflow Integration
- **Experiment Tracking**: Automatically creates and manages MLflow experiments
- **Parameter Logging**: Logs all model hyperparameters and training configuration
- **Metrics Tracking**: Records comprehensive evaluation metrics (MAE, MSE, RMSE, R²)
- **Model Registry**: Automatically registers trained models for deployment
- **Artifact Management**: Stores model files, feature information, and evaluation reports
- **Run Metadata**: Tags and organizes runs for easy filtering and comparison

### Enhanced Model Evaluation
- **Train/Test Split**: Proper evaluation with held-out test set (20% split)
- **Comprehensive Metrics**: Training and test metrics with overfitting detection
- **Model Signature**: Automatic inference of input/output schemas
- **Input Examples**: Stores sample inputs for model validation

### Backward Compatibility
- **Traditional Artifacts**: Still creates `model/model.pkl` and `model/model_features.json`
- **Same Data Pipeline**: Uses identical data loading and preprocessing logic
- **API Compatibility**: Generated models work with existing API infrastructure

## Usage

### Prerequisites
Ensure you have the conda environment activated:
```bash
conda activate housing
```

The environment already includes MLflow 3.1.4, so no additional installation is required.

### Running the Script
```bash
cd repo
python create_model_mlflow.py
```

### Expected Output
```
Loading data from data/kc_house_data.csv and data/zipcode_demographics.csv
Dataset loaded: 21613 samples, 33 features
Target statistics: mean=$540088.14, std=$367127.20
MLflow run started: <run_id>
Training model...
Training Metrics - MAE: $76122.37, RMSE: $143438.97, R²: 0.8425
Test Metrics - MAE: $101067.64, RMSE: $199631.95, R²: 0.7364

MLflow tracking completed!
Run ID: <run_id>
Experiment ID: <experiment_id>
Model URI: models:/<model_id>
Registered Model: real_estate_knn_model
```

## MLflow Tracking Details

### Logged Parameters
- **Dataset Information**: Size, feature count, target statistics
- **Model Parameters**: All KNeighborsRegressor hyperparameters
- **Preprocessing Parameters**: RobustScaler configuration
- **Training Configuration**: Test size, random state

### Logged Metrics
- **Training Metrics**: `train_mae`, `train_mse`, `train_rmse`, `train_r2`
- **Test Metrics**: `test_mae`, `test_mse`, `test_rmse`, `test_r2`
- **Overfitting Indicators**: `overfitting_ratio_mae`, `overfitting_ratio_r2`

### Logged Artifacts
- **Model**: Trained scikit-learn pipeline with signature
- **Feature Information**: JSON file with feature names and types
- **Evaluation Report**: Comprehensive training summary
- **Traditional Artifacts**: Backward-compatible pickle and JSON files

### Tags
- `model_type`: KNeighborsRegressor
- `preprocessing`: RobustScaler
- `problem_type`: regression
- `domain`: real_estate
- `data_source`: kc_house_data
- `author`: MLflow Enhanced Training
- `version`: 1.0

## Viewing Results

### Local MLflow UI
The script uses file-based tracking by default. To view results in the MLflow UI:

1. Start the MLflow UI server:
```bash
cd repo
mlflow ui --backend-store-uri file:./mlruns
```

2. Open your browser to: http://localhost:5000

3. Navigate to the "Real Estate Price Prediction" experiment

### MLflow Server Integration
To use with a remote MLflow server, update the `TRACKING_URI` in the script:
```python
TRACKING_URI = "http://your-mlflow-server:5000"
```

## Model Performance

The enhanced script provides detailed performance analysis:

- **Training Performance**: R² ≈ 0.84, MAE ≈ $76K
- **Test Performance**: R² ≈ 0.74, MAE ≈ $101K
- **Overfitting Analysis**: Automatic calculation of performance ratios

## Integration with Existing API

The script maintains full compatibility with the existing FastAPI service:
- Traditional model artifacts are still created in the `model/` directory
- Model format and feature list remain unchanged
- API can load models using existing logic

## Future Enhancements

This MLflow integration provides the foundation for:
- **A/B Testing**: Compare different model versions
- **Automated Retraining**: Schedule periodic model updates
- **Model Monitoring**: Track prediction performance over time
- **Experiment Comparison**: Systematic hyperparameter optimization
- **Deployment Automation**: Automated model promotion workflows

## Configuration

### Key Configuration Variables
```python
EXPERIMENT_NAME = "Real Estate Price Prediction"
TRACKING_URI = "file:./mlruns"  # Local file-based tracking
```

### Model Parameters
```python
model_params = {
    "n_neighbors": 5,
    "weights": "uniform",
    "algorithm": "auto",
    "leaf_size": 30,
    "p": 2,
    "metric": "minkowski",
}
```

### Preprocessing Parameters
```python
scaler_params = {
    "quantile_range": (25.0, 75.0),
    "with_centering": True,
    "with_scaling": True,
}
```

## Troubleshooting

### Common Warnings
- **Integer Column Warnings**: These are informational warnings about schema inference and don't affect functionality
- **Deprecated artifact_path**: This warning is expected and doesn't impact model logging

### File Permissions
Ensure the script has write permissions for:
- `./mlruns/` directory (MLflow tracking)
- `./model/` directory (traditional artifacts)
- Current directory (temporary artifact files)

## Comparison with Original Script

| Feature | create_model.py | create_model_mlflow.py |
|---------|----------------|------------------------|
| Model Training | ✅ | ✅ |
| Artifact Creation | ✅ | ✅ |
| Experiment Tracking | ❌ | ✅ |
| Metrics Logging | ❌ | ✅ |
| Model Registry | ❌ | ✅ |
| Performance Analysis | ❌ | ✅ |
| Test Set Evaluation | ❌ | ✅ |
| Overfitting Detection | ❌ | ✅ |
| Run Reproducibility | ❌ | ✅ |
| Model Versioning | ❌ | ✅ |

The MLflow-enhanced script provides all the functionality of the original while adding comprehensive experiment management capabilities essential for production ML workflows.
