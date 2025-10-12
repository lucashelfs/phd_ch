# Machine Learning Experiments

This directory contains all machine learning experimentation code and results for the Real Estate Price Prediction project.

## Directory Structure

```
experiments/
├── training_scripts/          # MLflow training and experimentation scripts
│   ├── create_model_mlflow.py                    # Basic MLflow KNN model training
│   ├── create_model_lightgbm_tuning.py          # LightGBM hyperparameter tuning
│   ├── create_model_xgboost_tuning.py           # XGBoost hyperparameter tuning
│   ├── model_comparison_data_experiments.py      # Comprehensive data pipeline experiments (162 experiments)
│   ├── optimized_hyperparameter_tuning.py       # Final optimized hyperparameter tuning (180 experiments)
│   └── compare_models.py                        # Model comparison utilities
└── results/                   # Experiment results and documentation
    ├── summaries/            # JSON experiment summaries
    │   ├── data_pipeline_experiments_summary.json
    │   ├── lightgbm_tuning_summary.json
    │   ├── xgboost_tuning_summary.json
    │   └── optimized_hyperparameter_tuning_summary.json
    └── reports/              # Markdown documentation and findings
        ├── HYPERPARAMETER_TUNING_FINDINGS.md
        ├── DATA_PIPELINE_EXPERIMENTS_FINDINGS.md
        └── OPTIMIZED_HYPERPARAMETER_TUNING_FINDINGS.md
```

## Experiment Timeline

### 1. Initial Hyperparameter Tuning
- **Script**: `create_model_lightgbm_tuning.py`, `create_model_xgboost_tuning.py`
- **Results**: R² = 0.7932 (LightGBM best)
- **Documentation**: `HYPERPARAMETER_TUNING_FINDINGS.md`

### 2. Data Pipeline Optimization
- **Script**: `model_comparison_data_experiments.py`
- **Experiments**: 162 comprehensive experiments
- **Results**: R² = 0.8038 (75/25 split + RobustScaler optimal)
- **Documentation**: `DATA_PIPELINE_EXPERIMENTS_FINDINGS.md`

### 3. Final Optimized Tuning
- **Script**: `optimized_hyperparameter_tuning.py`
- **Experiments**: 180 experiments with optimal data pipeline
- **Results**: R² = 0.8073 (Final champion model)
- **Documentation**: `OPTIMIZED_HYPERPARAMETER_TUNING_FINDINGS.md`

## Performance Evolution

| Stage | Model | R² Score | Improvement | Key Innovation |
|-------|-------|----------|-------------|----------------|
| Baseline | KNN | 0.7364 | - | Initial implementation |
| Initial Tuning | LightGBM | 0.7932 | +7.71% | Hyperparameter optimization |
| Data Pipeline Opt | LightGBM | 0.8038 | +9.15% | Optimal 75/25 split discovery |
| **Final Optimized** | **LightGBM** | **0.8073** | **+9.62%** | **Advanced parameter tuning** |

## Champion Model Configuration

**Final Best Model**: LightGBM with optimized hyperparameters
- **R² Score**: 0.8073 (80.73% variance explained)
- **MAE**: $90,182 average prediction error
- **Data Split**: 75/25 train/test
- **Preprocessing**: RobustScaler (25-75% quantiles)
- **Features**: Original 33 features (no engineering)
- **Outliers**: Kept for better performance

## Running Experiments

All training scripts are configured to work from the `experiments/training_scripts/` directory:

```bash
# Navigate to training scripts directory
cd experiments/training_scripts/

# Run individual experiments
python create_model_mlflow.py
python model_comparison_data_experiments.py
python optimized_hyperparameter_tuning.py

# View results in MLflow UI
mlflow ui --backend-store-uri file:../../mlruns
```

## MLflow Tracking

All experiments are tracked using MLflow with the tracking URI pointing to `../../mlruns` relative to the training scripts directory. This ensures all experiment data is centralized in the repository root's `mlruns/` directory.

### Experiment Names:
- `Real Estate Price Prediction` - Basic MLflow experiments
- `Real Estate Data Pipeline Experiments` - Comprehensive data pipeline testing
- `Real Estate Optimized Split Hyperparameter Tuning` - Final optimization experiments

## Key Findings

1. **LightGBM consistently outperformed** XGBoost and KNN across all experiments
2. **75/25 train/test split** provided optimal generalization vs. 80/20 or 70/30
3. **RobustScaler** outperformed StandardScaler and MinMaxScaler
4. **Original features** performed better than engineered features (polynomial, log transforms)
5. **Keeping outliers** improved R² performance despite higher MAE on outlier-removed data
6. **Moderate ensemble sizes** (75-100 estimators) were optimal for LightGBM
7. **Deep trees with controlled leaves** (depth=15, leaves=50) prevented overfitting

## Production Deployment

The champion model configuration is production-ready with:
- Excellent generalization (overfitting ratio: 1.095)
- Consistent performance across multiple runs
- Complete MLflow artifact tracking
- Comprehensive documentation and reproducibility information

For production deployment, use the champion model parameters documented in `OPTIMIZED_HYPERPARAMETER_TUNING_FINDINGS.md`.
