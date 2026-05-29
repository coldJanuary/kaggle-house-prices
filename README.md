# Kaggle — House Prices: Advanced Regression

**Competition:** [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)  
**Result:** Top 7% on public leaderboard (RMSE = 0.11482)

## Approach

1. **EDA & feature engineering** — handle missing values, log-transform skewed features, encode categoricals, add interaction features (total area, house age, remodel age)
2. **Modelling** — CatBoost + LightGBM trained independently, blended with optimised weights
3. **Interpretation** — SHAP values to explain which features drive price predictions

## Results

| Model | CV RMSE | LB RMSE |
|-------|---------|---------|
| CatBoost | 0.1193 | 0.1201 |
| LightGBM | 0.1187 | 0.1196 |
| **Blend (0.45 / 0.55)** | **0.1161** | **0.1148** |

## Structure

```
kaggle-house-prices/
├── notebooks/
│   ├── 01_eda.ipynb            # exploratory analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_modelling.ipynb      # training, blending, SHAP
├── src/
│   ├── features.py             # feature engineering pipeline
│   └── train.py                # training loop with cross-validation
├── requirements.txt
└── README.md
```

## Stack

`Python` `pandas` `NumPy` `CatBoost` `LightGBM` `scikit-learn` `SHAP` `Optuna` `matplotlib` `seaborn`
