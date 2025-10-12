# Real Estate Price Prediction - Model Comparison

This document summarizes the comparison between three different machine learning models for real estate price prediction: K-Nearest Neighbors (KNN), XGBoost, and LightGBM.

## Model Performance Summary

Based on the MLflow experiment tracking, here are the results from our model comparison:

### 🏆 Overall Rankings

1. **LightGBM** (Recommended)
   - Test R²: 0.7867
   - Test MAE: $92,569.94
   - Test RMSE: $179,561.07
   - Overfitting Ratio: 1.0899 (Best generalization)

2. **XGBoost** 
   - Test R²: 0.7714
   - Test MAE: $92,359.06 (Lowest MAE)
   - Test RMSE: $185,912.15
   - Overfitting Ratio: 1.1749

3. **K-Nearest Neighbors**
   - Test R²: 0.7364
   - Test MAE: $101,067.64
   - Test RMSE: $199,631.95
   - Overfitting Ratio: 1.1441

## Key Insights

### Model Strengths

**LightGBM:**
- Highest R² score (78.67% variance explained)
- Best generalization (lowest overfitting ratio)
- Good balance between accuracy and generalization
- Fast training time

**XGBoost:**
- Lowest Mean Absolute Error ($92,359)
- Highest training accuracy (R² = 0.9063)
- Strong predictive performance
- Robust to outliers

**K-Nearest Neighbors:**
- Simple, interpretable model
- No assumptions about data distribution
- Good baseline performance
- Easy to understand and explain

### Performance Analysis

- **Best Overall Model**: LightGBM wins with the highest composite score (0.9993)
- **Most Accurate Predictions**: XGBoost has the lowest MAE by $210
- **Best Generalization**: LightGBM shows the least overfitting
- **Prediction Range**: All models predict within ~$8,500 MAE range

## Technical Implementation

### Consistent Data Pipeline
All models use identical:
- Data loading and preprocessing
- Train/test split (80/20, random_state=42)
- Feature scaling (RobustScaler with 25-75% quantile range)
- Evaluation metrics (MAE, MSE, RMSE, R²)

### Model Configurations

**XGBoost Parameters:**
```python
{
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "objective": "reg:squarederror"
}
```

**LightGBM Parameters:**
```python
{
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "objective": "regression"
}
```

**KNN Parameters:**
```python
{
    "n_neighbors": 5,
    "weights": "uniform",
    "algorithm": "auto",
    "metric": "minkowski"
}
```

## MLflow Integration

### Experiment Tracking
All models are tracked in the same MLflow experiment: "Real Estate Price Prediction"

### Logged Information
- **Parameters**: All hyperparameters and preprocessing settings
- **Metrics**: Training and test performance metrics
- **Artifacts**: Model files, feature information, evaluation reports
- **Tags**: Model type, preprocessing, domain, version
- **Models**: Registered in MLflow Model Registry

### Model Registry
- `real_estate_knn_model`
- `real_estate_xgboost_model`
- `real_estate_lightgbm_model`

## Usage Instructions

### Running Individual Models
```bash
# Activate conda environment
conda activate housing

# Train KNN model
python create_model_mlflow.py

# Train XGBoost model
python create_model_xgboost.py

# Train LightGBM model
python create_model_lightgbm.py
```

### Comparing Models
```bash
# Run model comparison
python compare_models.py
```

### Viewing Results in MLflow UI
```bash
# Start MLflow UI
mlflow ui --backend-store-uri file:./mlruns

# Open browser to http://localhost:5000
# Navigate to "Real Estate Price Prediction" experiment
```

## Business Impact

### Prediction Accuracy
- All models achieve 73-79% variance explanation (R²)
- Prediction errors range from $92K-$101K MAE
- Suitable for real estate valuation with appropriate confidence intervals

### Model Selection Recommendation

**For Production Use: LightGBM**
- Best overall performance with good generalization
- Balanced accuracy and robustness
- Fast inference time
- Lower risk of overfitting

**For Highest Accuracy: XGBoost**
- Lowest prediction error in dollars
- Excellent for scenarios where minimizing MAE is critical
- Slightly more prone to overfitting

**For Interpretability: KNN**
- Easiest to explain to stakeholders
- Good baseline performance
- Transparent prediction logic

## Future Improvements

### Model Enhancement
1. **Hyperparameter Tuning**: Use MLflow with Optuna for systematic optimization
2. **Feature Engineering**: Add polynomial features, interaction terms
3. **Ensemble Methods**: Combine top models for better performance
4. **Cross-Validation**: Implement k-fold CV for more robust evaluation

### MLflow Integration
1. **A/B Testing**: Compare model versions in production
2. **Model Monitoring**: Track prediction drift and performance
3. **Automated Retraining**: Schedule periodic model updates
4. **Model Serving**: Deploy models with MLflow Model Serving

### Data Pipeline
1. **Feature Selection**: Use feature importance from tree models
2. **Data Quality**: Implement data validation and monitoring
3. **Real-time Features**: Add market trends and economic indicators
4. **Geographic Features**: Enhance location-based features

## Conclusion

The model comparison demonstrates that gradient boosting methods (XGBoost and LightGBM) significantly outperform the traditional KNN approach for real estate price prediction. LightGBM emerges as the recommended model due to its superior balance of accuracy, generalization, and computational efficiency.

The MLflow integration provides a solid foundation for model lifecycle management, enabling easy comparison, versioning, and deployment of models in production environments.
