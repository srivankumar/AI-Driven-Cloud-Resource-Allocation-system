import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


TRAIN_ROWS = 20_000
TARGET_COL = "Resource_Allocation"
BASE_NUMERIC_COLS = [
    "cpu_utilization",
    "memory_usage",
    "storage_usage",
    "workload",
]
LAG_STEPS = [1, 2, 3, 6, 12, 24]
ROLLING_WINDOWS = [3, 6, 12, 24]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [c.strip().replace(" ", "_") for c in normalized.columns]
    return normalized


def build_feature_frame(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = normalize_columns(raw_df)

    required = {"timestamp", *BASE_NUMERIC_COLS, TARGET_COL}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["row_number"] = np.arange(1, len(df) + 1)

    for col in BASE_NUMERIC_COLS + [TARGET_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Deterministic missing-value treatment for predictors.
    for col in BASE_NUMERIC_COLS:
        df[col] = df[col].ffill().bfill()
        if df[col].isna().any():
            fallback = float(df[col].median()) if not pd.isna(df[col].median()) else 0.0
            df[col] = df[col].fillna(fallback)

    feats = pd.DataFrame(index=df.index)

    feats["hour"] = df["timestamp"].dt.hour
    feats["day"] = df["timestamp"].dt.day
    feats["month"] = df["timestamp"].dt.month

    cyclical = {"hour": 24, "day": 31, "month": 12}
    for key, period in cyclical.items():
        angle = 2.0 * np.pi * feats[key] / period
        feats[f"{key}_sin"] = np.sin(angle)
        feats[f"{key}_cos"] = np.cos(angle)

    for col in BASE_NUMERIC_COLS:
        series = df[col]
        for lag in LAG_STEPS:
            feats[f"{col}_lag_{lag}"] = series.shift(lag)

    for col in BASE_NUMERIC_COLS:
        shifted = df[col].shift(1)
        for window in ROLLING_WINDOWS:
            roll = shifted.rolling(window=window)
            feats[f"{col}_roll_mean_{window}"] = roll.mean()
            feats[f"{col}_roll_std_{window}"] = roll.std()
            feats[f"{col}_roll_min_{window}"] = roll.min()
            feats[f"{col}_roll_max_{window}"] = roll.max()

    feats["cpu_mem_ratio"] = df["cpu_utilization"] / (df["memory_usage"] + 1e-6)
    feats["cpu_workload"] = df["cpu_utilization"] * df["workload"]
    feats["mem_workload"] = df["memory_usage"] * df["workload"]

    output = pd.concat(
        [
            df[["timestamp", "row_number", TARGET_COL, *BASE_NUMERIC_COLS]],
            feats,
        ],
        axis=1,
    )
    output = output.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    feature_cols = [
        c
        for c in output.columns
        if c not in {"timestamp", TARGET_COL, "row_number"}
    ]
    return output, feature_cols


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sequential XGB + residual SGD models.")
    parser.add_argument("--data", default="dataset.csv", help="Path to dataset CSV")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path.resolve()}")

    raw_df = pd.read_csv(data_path)
    feature_df, feature_cols = build_feature_frame(raw_df)

    train_df = feature_df[feature_df["row_number"] <= TRAIN_ROWS].copy()
    if train_df.empty:
        raise ValueError("No training rows available after feature engineering and NaN drop.")

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]

    xgb = XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.01,
        reg_lambda=1.0,
        random_state=42,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    residuals = y_train.values - xgb.predict(X_train)
    sgd = SGDRegressor(
        loss="squared_error",
        penalty="l2",
        alpha=1e-4,
        learning_rate="invscaling",
        eta0=0.01,
        power_t=0.25,
        max_iter=2000,
        tol=1e-4,
        random_state=42,
    )
    sgd.fit(X_train_scaled, residuals)

    with open("xgb.pkl", "wb") as f:
        pickle.dump(xgb, f)
    with open("sgd.pkl", "wb") as f:
        pickle.dump(sgd, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("features.pkl", "wb") as f:
        pickle.dump(feature_cols, f)

    print(f"Training complete. Saved models with {len(feature_cols)} engineered features.")
    print(f"Training samples used: {len(train_df)} (from first {TRAIN_ROWS} sequential rows)")


if __name__ == "__main__":
    main()
