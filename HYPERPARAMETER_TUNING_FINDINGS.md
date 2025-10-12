# Hyperparameter Tuning Results - First Grid Search

## Overview
Initial hyperparameter optimization conducted on baseline models using 80/20 train/test split with RobustScaler preprocessing.

## Results Summary

### Model Performance Rankings
1. **LightGBM (Tuned)** - R²: 0.7932, MAE: $92,789 ⭐ **BEST**
2. **LightGBM (Baseline)** - R²: 0.7867, MAE: $92,570
3. **XGBoost (Tuned)** - R²: 0.7714, MAE: $93,903
4. **XGBoost (Baseline)** - R²: 0.7714, MAE: $92,359
5. **KNN (Baseline)** - R²: 0.7364, MAE: $101,068

### Key Improvements
- **LightGBM**: +0.65% R² improvement (0.7867 → 0.7932)
- **Better generalization**: Overfitting ratio improved from 1.0899 to 1.0724
- **XGBoost**: No significant improvement over baseline

## Optimal Parameters Found

### LightGBM Best Configuration
```python
{
    "n_estimators": 50,
    "max_depth": 9,
    "learning_rate": 0.1,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 1.5,
    "num_leaves": 31
}
```

### XGBoost Best Configuration
```python
{
    "n_estimators": 50,
    "max_depth": 6,
    "learning_rate": 0.2,
    "subsample": 1.0,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0
}
```

## Key Insights

### Model Architecture
- **Smaller models perform better**: 50 estimators often outperformed 100-200
- **Learning rates**: 0.1-0.2 range optimal for both models
- **Regularization**: Light regularization (α=0.1-1.0) helps prevent overfitting

### Algorithm-Specific Findings
- **LightGBM**: Benefits from deeper trees (depth=9) with controlled leaves (31)
- **XGBoost**: Prefers moderate depth (6) with higher learning rates (0.2)
- **Feature sampling**: 80-90% column sampling works well for both

## Limitations of Current Approach
- **Single data split**: 80/20 split may not be optimal
- **Basic preprocessing**: Only RobustScaler used
- **No feature engineering**: Missing interactions, polynomials
- **Limited search space**: 50 combinations per model

## Next Steps
- Test different train/test splits (75/25, 70/30)
- Experiment with preprocessing techniques
- Explore feature engineering approaches
- Implement cross-validation for more robust estimates
