import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import shap
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(np.clip(y_pred, 0, None))))


def cross_val_predict(model, X, y, n_splits=5, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(len(y))
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        model.fit(X_tr, y_tr)
        oof[val_idx] = model.predict(X_val)
        print(f"  Fold {fold+1} RMSLE: {rmsle(y_val, oof[val_idx]):.5f}")
    print(f"  OOF RMSLE: {rmsle(y, oof):.5f}")
    return oof


def train_catboost(X, y, cat_features):
    model = CatBoostRegressor(
        iterations=2000, learning_rate=0.03, depth=7,
        loss_function="RMSE", eval_metric="RMSE",
        random_seed=42, verbose=0,
    )
    return model, cross_val_predict(model.fit(X, y, cat_features=cat_features), X, y)


def train_lgbm(X, y):
    model = LGBMRegressor(
        n_estimators=2000, learning_rate=0.03, num_leaves=63,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1,
    )
    return model, cross_val_predict(model, X, y)


def blend_predictions(preds_list, weights):
    weights = np.array(weights) / sum(weights)
    return sum(p * w for p, w in zip(preds_list, weights))


def explain_with_shap(model, X, n_samples=500):
    explainer = shap.TreeExplainer(model)
    sample = X.sample(min(n_samples, len(X)), random_state=42)
    shap_values = explainer.shap_values(sample)
    shap.summary_plot(shap_values, sample, plot_type="bar", max_display=20)
    return shap_values
