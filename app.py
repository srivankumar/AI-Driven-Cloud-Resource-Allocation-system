import pickle
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template


TRAIN_ROWS = 20_000
TEST_END_ROW = 26_000  # inclusive in business terms
TARGET_COL = "Resource_Allocation"
BASE_NUMERIC_COLS = [
    "cpu_utilization",
    "memory_usage",
    "storage_usage",
    "workload",
]
LAG_STEPS = [1, 2, 3, 6, 12, 24]
ROLLING_WINDOWS = [3, 6, 12, 24]

app = Flask(__name__)
state_lock = threading.Lock()


xgb_model = None
sgd_model = None
scaler = None
feature_cols = None
test_df = None
current_pointer = 0


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

    derived_feature_cols = [
        c
        for c in output.columns
        if c not in {"timestamp", TARGET_COL, "row_number"}
    ]
    return output, derived_feature_cols


def align_feature_vector(row_df: pd.DataFrame, ordered_features: list[str]) -> pd.DataFrame:
    aligned = row_df.copy()
    missing = [col for col in ordered_features if col not in aligned.columns]
    for col in missing:
        aligned[col] = 0.0
    return aligned[ordered_features]


def load_artifacts() -> None:
    global xgb_model, sgd_model, scaler, feature_cols, test_df

    for file_name in ["xgb.pkl", "sgd.pkl", "scaler.pkl", "features.pkl"]:
        if not Path(file_name).exists():
            raise FileNotFoundError(f"Missing artifact: {file_name}. Run train.py first.")

    with open("xgb.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open("sgd.pkl", "rb") as f:
        sgd_model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("features.pkl", "rb") as f:
        feature_cols = pickle.load(f)

    full_raw = pd.read_csv("dataset.csv")
    engineered_df, _ = build_feature_frame(full_raw)

    # Apply feature engineering first, then sequential split rows 20001-26000.
    test_df = engineered_df[
        (engineered_df["row_number"] > TRAIN_ROWS)
        & (engineered_df["row_number"] <= TEST_END_ROW)
    ].reset_index(drop=True)

    if test_df.empty:
        raise ValueError("No test rows available for split range 20001-26000.")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict_next", methods=["GET"])
def predict_next():
    global current_pointer, sgd_model

    with state_lock:
        if current_pointer >= len(test_df):
            return jsonify(
                {
                    "done": True,
                    "message": "No more rows available in the sequential test window.",
                    "processed": int(current_pointer),
                    "total": int(len(test_df)),
                }
            )

        row = test_df.iloc[current_pointer]
        row_frame = pd.DataFrame([row])
        X_row = align_feature_vector(row_frame, feature_cols)

        xgb_pred = float(xgb_model.predict(X_row)[0])
        X_row_scaled = scaler.transform(X_row)
        sgd_correction = float(sgd_model.predict(X_row_scaled)[0])
        final_pred = xgb_pred + sgd_correction

        actual = float(row[TARGET_COL])
        error = actual - final_pred

        residual_target = np.array([actual - xgb_pred], dtype=float)
        sgd_model.partial_fit(X_row_scaled, residual_target)

        predicted_servers = max(1, int(np.ceil(final_pred / 10.0)))
        actual_servers = max(1, int(np.ceil(actual / 10.0)))
        warm_servers = 4

        response = {
            "done": False,
            "time": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            "predicted": round(final_pred, 6),
            "actual": round(actual, 6),
            "error": round(error, 6),
            "predicted_servers": int(predicted_servers),
            "actual_servers": int(actual_servers),
            "warm_servers": int(warm_servers),
            "step": int(current_pointer + 1),
            "remaining": int(len(test_df) - current_pointer - 1),
        }

        current_pointer += 1
        return jsonify(response)


if __name__ == "__main__":
    load_artifacts()
    app.run(host="0.0.0.0", port=5000, debug=False)
else:
    load_artifacts()
