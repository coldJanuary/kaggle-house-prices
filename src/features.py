import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


COLS_WITH_NA_MEANING = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
    "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish",
    "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature",
]

ORDINAL_MAP = {
    "ExterQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "ExterCond": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtQual":  {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "KitchenQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageQual": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
}


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in COLS_WITH_NA_MEANING:
        if col in df.columns:
            df[col] = df[col].fillna("None")
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalPorchSF"] = (df["OpenPorchSF"] + df["EnclosedPorch"]
                          + df["3SsnPorch"] + df["ScreenPorch"])
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodelAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["GarageAge"] = df["YrSold"] - df["GarageYrBlt"].replace(0, df["YearBuilt"])
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["TotalBath"] = (df["FullBath"] + 0.5 * df["HalfBath"]
                       + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])
    df["OverallScore"] = df["OverallQual"] * df["OverallCond"]
    return df


def encode_ordinals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, mapping in ORDINAL_MAP.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = fill_missing(df)
    df = add_features(df)
    df = encode_ordinals(df)
    return df
