# Repository Walkthrough Guide

This guide provides a comprehensive walkthrough for newcomers to understand and utilize the real estate price prediction system. The repository implements a machine learning pipeline with MLflow experiment tracking, containerized API deployment, and model versioning capabilities.

## Architecture Overview

The system consists of:
- **Machine Learning Pipeline**: Model training with MLflow tracking
- **REST API**: FastAPI-based prediction service with V1 (pickle) and V2 (MLflow) endpoints
- **Storage Backend**: PostgreSQL for experiment metadata, MinIO for model artifacts
- **Containerization**: Docker Compose orchestration for all services

## Prerequisites

- Conda package manager
- Docker and Docker Compose
- Git

## 1. Environment Setup

### Install Conda Environment

Create and activate the conda environment from the provided specification:

```bash
conda env create -f conda_environment.yml
conda activate housing
```

The environment includes:
- Python 3.9
- pandas 2.1.1
- scikit-learn 1.3.1
- MLflow 3.1.4
- XGBoost 2.0.3
- LightGBM 4.1.0

### Verify Installation

```bash
python --version
conda list | grep -E "(pandas|scikit-learn|mlflow)"
```

## 2. Initial Model Generation

### Run the Original Model Script

Execute the baseline model creation script:

```bash
python create_model.py
```

This script:
- Loads housing data from `data/kc_house_data.csv`
- Merges with demographic data from `data/zipcode_demographics.csv`
- Trains a K-Nearest Neighbors regression model
- Outputs artifacts to the `model/` directory:
  - `model.pkl`: Serialized model
  - `model_features.json`: Feature list in training order

### Verify Model Artifacts

```bash
ls -la model/
cat model/model_features.json
```

## 3. API Testing with Docker

### Start the Complete Stack

Launch all services using Docker Compose:

```bash
docker-compose up -d
```

This starts:
- **PostgreSQL** (port 5432): Experiment metadata storage
- **MinIO** (ports 9000, 9001): Artifact storage with S3-compatible API
- **MLflow Server** (port 5000): Experiment tracking and model registry
- **Real Estate API** (port 8000): Prediction service

### Verify Service Health

Check all services are running:

```bash
docker-compose ps
```

Test API availability:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/versions
```

### Test API Endpoints

**V1 Endpoints (Pickle-based models):**
```bash
# Get API information
curl http://localhost:8000/v1/info

# Test prediction with full features
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bedrooms": 3,
    "bathrooms": 2.5,
    "sqft_living": 2000,
    "sqft_lot": 8000,
    "floors": 2,
    "sqft_above": 1500,
    "sqft_basement": 500,
    "zipcode": "98103"
  }'

# Test minimal prediction
curl -X POST http://localhost:8000/v1/predict/minimal \
  -H "Content-Type: application/json" \
  -d '{
    "bedrooms": 3,
    "bathrooms": 2.5,
    "sqft_living": 2000,
    "zipcode": "98103"
  }'
```

**V2 Endpoints (MLflow-based models):**
```bash
# Get V2 API information
curl http://localhost:8000/v2/info

# Test V2 prediction (same payload format)
curl -X POST http://localhost:8000/v2/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bedrooms": 3,
    "bathrooms": 2.5,
    "sqft_living": 2000,
    "sqft_lot": 8000,
    "floors": 2,
    "sqft_above": 1500,
    "sqft_basement": 500,
    "zipcode": "98103"
  }'
```

## 4. MLflow Experiment Management

### Access MLflow UI

Navigate to the MLflow interface:

```bash
open http://localhost:5000
```

The MLflow server provides:
- **Experiment Tracking**: Run history, parameters, metrics
- **Model Registry**: Versioned model storage and lifecycle management
- **Artifact Storage**: Model files, plots, and other outputs

### Storage Architecture

**PostgreSQL Backend**: Stores experiment metadata including:
- Run information and parameters
- Metrics and tags
- Model registry metadata

**MinIO Artifact Store**: S3-compatible storage for:
- Model artifacts (pickle files, MLflow models)
- Training plots and visualizations
- Custom artifacts and logs

Access MinIO console at: http://localhost:9001
- Username: `minio`
- Password: `minio123`

## 5. Model Development Workflow

*[PLACEHOLDER SECTION]*

This section will cover:
- Creating new experiments in MLflow
- Training models with different algorithms (LightGBM, XGBoost)
- Hyperparameter tuning and cross-validation
- Model evaluation and comparison
- Registering champion models

Available training scripts:
- `create_model_lightgbm.py`
- `create_model_xgboost.py`
- `train_with_docker_mlflow.py`

## 6. Model Deployment Process

### Configure Champion Model

To deploy a new model from MLflow Model Registry:

1. **Identify the Model URI**: From MLflow UI, note the model URI (e.g., `models:/model_name/version`)

2. **Update Environment Configuration**: Edit the `.env` file:

```bash
# Update the champion model URI
CHAMPION_MODEL_MLFLOW_URI="models:/your_model_name/latest"
```

Example configurations:
```bash
# Use latest version of a specific model
CHAMPION_MODEL_MLFLOW_URI="models:/lightgbm_house_price_model/latest"

# Use specific version
CHAMPION_MODEL_MLFLOW_URI="models:/lightgbm_house_price_model/3"

# Use model by stage
CHAMPION_MODEL_MLFLOW_URI="models:/lightgbm_house_price_model/Production"
```

### Reload Application

Restart the API service to load the new model:

```bash
docker-compose restart real-estate-api
```

### Verify Model Deployment

Check that the new model is active:

```bash
# Verify V2 API is enabled
curl http://localhost:8000/versions

# Get model information
curl http://localhost:8000/v2/info

# Test prediction with new model
curl -X POST http://localhost:8000/v2/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bedrooms": 4,
    "bathrooms": 3,
    "sqft_living": 2500,
    "sqft_lot": 10000,
    "floors": 2,
    "sqft_above": 2000,
    "sqft_basement": 500,
    "zipcode": "98105"
  }'
```

The new model is now available through V2 endpoints while V1 continues serving the original pickle-based model.

## 7. Model Comparison Framework

*[PLACEHOLDER SECTION]*

This section will cover:
- Comparing baseline (V1) vs. champion (V2) model performance
- A/B testing methodologies
- Performance metrics evaluation
- Statistical significance testing
- Model drift detection

Available comparison tools:
- `experiments/model_performance_comparison.py`
- Performance visualization scripts
- Automated evaluation pipelines

## API Versioning Strategy

The system implements a dual-API approach:

**V1 API (Always Available)**:
- Uses pickle-serialized models from `model/` directory
- Backward compatible with existing integrations
- Endpoints: `/v1/predict`, `/v1/predict/minimal`, `/v1/info`

**V2 API (Conditional)**:
- Uses MLflow Model Registry models
- Enabled only when `CHAMPION_MODEL_MLFLOW_URI` is configured
- Endpoints: `/v2/predict`, `/v2/predict/minimal`, `/v2/info`
- Same request/response format as V1 for easy migration

## Troubleshooting

### Common Issues

**V2 API Returns 404**:
- Verify `CHAMPION_MODEL_MLFLOW_URI` is set in `.env`
- Check MLflow model exists and is accessible
- Restart the API service

**Model Loading Errors**:
- Verify MLflow server is running and accessible
- Check MinIO connectivity and credentials
- Review API container logs: `docker-compose logs real-estate-api`

**Service Health Issues**:
- Check all containers are running: `docker-compose ps`
- Verify port availability (5000, 8000, 9000, 5432)
- Review service logs: `docker-compose logs [service-name]`

### Useful Commands

```bash
# View all service logs
docker-compose logs -f

# Restart specific service
docker-compose restart [service-name]

# Check API health
curl http://localhost:8000/health

# View available API versions
curl http://localhost:8000/versions

# Stop all services
docker-compose down

# Clean restart
docker-compose down && docker-compose up -d
```

## Next Steps

1. Explore the MLflow UI to understand experiment tracking
2. Run model training scripts to create new experiments
3. Register models in the MLflow Model Registry
4. Deploy and test new models using the V2 API
5. Implement model comparison and evaluation workflows

For detailed implementation examples, refer to the `examples/` directory and existing test scripts in `tests/`.
