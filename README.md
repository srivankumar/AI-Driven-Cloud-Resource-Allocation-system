# AI-Driven Cloud Resource Allocation System

An intelligent machine learning system for predicting optimal cloud resource allocation using XGBoost and residual SGD models.

## 📋 Project Overview

This system predicts the optimal number of servers and resource allocation needed based on real-time metrics like CPU utilization, memory usage, storage usage, and workload. It uses an ensemble approach combining XGBoost with online learning via SGD to adapt to changing patterns.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Flask Web Application                   │
│              (Real-time Predictions & UI)                │
└─────────────────────────────────────────────────────────┘
                          ↓
        ┌─────────────────┴─────────────────┐
        ↓                                     ↓
   ┌──────────────┐              ┌──────────────────┐
   │ XGBoost Model│              │  SGD Regressor   │
   │  (Primary)   │              │  (Residual)      │
   └──────────────┘              └──────────────────┘
        ↓                                     ↓
   ┌────────────────────────────────────────────────┐
   │     Feature Engineering Pipeline                │
   │  • Temporal Features (hour, day, month)        │
   │  • Cyclical Encoding (sin/cos)                 │
   │  • Lag Features (1, 2, 3, 6, 12, 24 steps)    │
   │  • Rolling Statistics (mean, std, min, max)   │
   │  • Interaction Features (ratios & products)    │
   └────────────────────────────────────────────────┘
        ↓
   ┌────────────────────────────────────────────────┐
   │            Input Dataset (CSV)                  │
   │  • Timestamp                                    │
   │  • CPU Utilization, Memory, Storage, Workload  │
   │  • Resource Allocation (Target)                 │
   └────────────────────────────────────────────────┘
```

## 📊 Dataset

**Column Structure:**
```
timestamp,cpu_utilization,memory_usage,storage_usage,workload,Resource_Allocation
2024-01-01 00:00:00,45.2,52.1,60.5,0.7,580
2024-01-01 01:00:00,48.3,55.2,61.2,0.75,595
...
```

**Data Split:**
- **Training**: Rows 1-20,000
- **Testing**: Rows 20,001-26,000

## 🚀 Quick Start

### Prerequisites

```bash
pip install flask pandas numpy scikit-learn xgboost
```

### Installation

```bash
# Clone repository
git clone https://github.com/srivankumar/AI-Driven-Cloud-Resource-Allocation-system.git
cd AI-Driven-Cloud-Resource-Allocation-system

# Install dependencies
pip install -r requirements.txt
```

### Training the Models

```bash
python train.py --data dataset.csv
```

This generates:
- `xgb.pkl` - Trained XGBoost model
- `sgd.pkl` - Trained SGD regressor
- `scaler.pkl` - StandardScaler for SGD input
- `features.pkl` - Feature column ordering

### Running the Application

```bash
python app.py
```

Access the web interface at: `http://localhost:5000`

## 📁 Project Structure

```
AI-Driven-Cloud-Resource-Allocation-system/
├── app.py                          # Flask web application
├── train.py                        # Model training pipeline
├── dataset.csv                     # Training & test data
├── templates/
│   └── index.html                  # Web UI
├── Research papers/                # Reference materials
└── README.md                       # This file
```

## 💻 Code Examples

### 1. Training Pipeline (train.py)

```python
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Model Configuration
TRAIN_ROWS = 20_000
TARGET_COL = "Resource_Allocation"
BASE_NUMERIC_COLS = [
    "cpu_utilization",
    "memory_usage",
    "storage_usage",
    "workload",
]

# Feature Engineering Configuration
LAG_STEPS = [1, 2, 3, 6, 12, 24]
ROLLING_WINDOWS = [3, 6, 12, 24]

def build_feature_frame(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Engineer features from raw resource metrics.
    
    Returns:
        Tuple of (engineered_df, feature_column_names)
    """
    df = normalize_columns(raw_df)
    
    # Validate required columns
    required = {"timestamp", *BASE_NUMERIC_COLS, TARGET_COL}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    
    # Parse timestamp and sort
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["row_number"] = np.arange(1, len(df) + 1)
    
    # Convert to numeric & handle missing values
    for col in BASE_NUMERIC_COLS + [TARGET_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    for col in BASE_NUMERIC_COLS:
        df[col] = df[col].ffill().bfill()
        if df[col].isna().any():
            fallback = float(df[col].median()) if not pd.isna(df[col].median()) else 0.0
            df[col] = df[col].fillna(fallback)
    
    feats = pd.DataFrame(index=df.index)
    
    # Temporal Features
    feats["hour"] = df["timestamp"].dt.hour
    feats["day"] = df["timestamp"].dt.day
    feats["month"] = df["timestamp"].dt.month
    
    # Cyclical Encoding (sin/cos transformation)
    cyclical = {"hour": 24, "day": 31, "month": 12}
    for key, period in cyclical.items():
        angle = 2.0 * np.pi * feats[key] / period
        feats[f"{key}_sin"] = np.sin(angle)
        feats[f"{key}_cos"] = np.cos(angle)
    
    # Lag Features
    for col in BASE_NUMERIC_COLS:
        series = df[col]
        for lag in LAG_STEPS:
            feats[f"{col}_lag_{lag}"] = series.shift(lag)
    
    # Rolling Statistics
    for col in BASE_NUMERIC_COLS:
        shifted = df[col].shift(1)
        for window in ROLLING_WINDOWS:
            roll = shifted.rolling(window=window)
            feats[f"{col}_roll_mean_{window}"] = roll.mean()
            feats[f"{col}_roll_std_{window}"] = roll.std()
            feats[f"{col}_roll_min_{window}"] = roll.min()
            feats[f"{col}_roll_max_{window}"] = roll.max()
    
    # Interaction Features
    feats["cpu_mem_ratio"] = df["cpu_utilization"] / (df["memory_usage"] + 1e-6)
    feats["cpu_workload"] = df["cpu_utilization"] * df["workload"]
    feats["mem_workload"] = df["memory_usage"] * df["workload"]
    
    # Combine and clean
    output = pd.concat(
        [
            df[["timestamp", "row_number", TARGET_COL, *BASE_NUMERIC_COLS]],
            feats,
        ],
        axis=1,
    )
    output = output.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    
    feature_cols = [c for c in output.columns if c not in {"timestamp", TARGET_COL, "row_number"}]
    return output, feature_cols


def main() -> None:
    """Train XGBoost + residual SGD ensemble."""
    
    # Load data
    raw_df = pd.read_csv("dataset.csv")
    feature_df, feature_cols = build_feature_frame(raw_df)
    
    # Split data
    train_df = feature_df[feature_df["row_number"] <= TRAIN_ROWS].copy()
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    
    # Train XGBoost
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
    
    # Prepare SGD for residual learning
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    residuals = y_train.values - xgb.predict(X_train)
    
    # Train SGD on residuals
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
    
    # Save artifacts
    with open("xgb.pkl", "wb") as f:
        pickle.dump(xgb, f)
    with open("sgd.pkl", "wb") as f:
        pickle.dump(sgd, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("features.pkl", "wb") as f:
        pickle.dump(feature_cols, f)
    
    print(f"Training complete. Saved models with {len(feature_cols)} engineered features.")


if __name__ == "__main__":
    main()
```

### 2. Flask Web Application (app.py)

```python
import pickle
import threading
from pathlib import Path
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template

TRAIN_ROWS = 20_000
TEST_END_ROW = 26_000
TARGET_COL = "Resource_Allocation"
BASE_NUMERIC_COLS = ["cpu_utilization", "memory_usage", "storage_usage", "workload"]

app = Flask(__name__)
state_lock = threading.Lock()

# Global state
xgb_model = None
sgd_model = None
scaler = None
feature_cols = None
test_df = None
current_pointer = 0


def load_artifacts() -> None:
    """Load trained models and preprocessing objects."""
    global xgb_model, sgd_model, scaler, feature_cols, test_df
    
    required_files = ["xgb.pkl", "sgd.pkl", "scaler.pkl", "features.pkl"]
    for file_name in required_files:
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
    
    # Load test data
    full_raw = pd.read_csv("dataset.csv")
    engineered_df, _ = build_feature_frame(full_raw)
    
    test_df = engineered_df[
        (engineered_df["row_number"] > TRAIN_ROWS)
        & (engineered_df["row_number"] <= TEST_END_ROW)
    ].reset_index(drop=True)
    
    if test_df.empty:
        raise ValueError("No test rows available for split range 20001-26000.")


@app.route("/")
def home():
    """Serve the main web interface."""
    return render_template("index.html")


@app.route("/predict_next", methods=["GET"])
def predict_next():
    """
    Sequential prediction endpoint.
    
    Predicts resource allocation for the next test row and updates the SGD model
    (online learning).
    
    Returns:
        JSON with prediction results, actual values, and performance metrics
    """
    global current_pointer, sgd_model
    
    with state_lock:
        # Check if all test samples processed
        if current_pointer >= len(test_df):
            return jsonify({
                "done": True,
                "message": "No more rows available in the sequential test window.",
                "processed": int(current_pointer),
                "total": int(len(test_df)),
            })
        
        # Get current row
        row = test_df.iloc[current_pointer]
        row_frame = pd.DataFrame([row])
        X_row = align_feature_vector(row_frame, feature_cols)
        
        # XGBoost prediction
        xgb_pred = float(xgb_model.predict(X_row)[0])
        
        # SGD correction
        X_row_scaled = scaler.transform(X_row)
        sgd_correction = float(sgd_model.predict(X_row_scaled)[0])
        
        # Final ensemble prediction
        final_pred = xgb_pred + sgd_correction
        
        # Calculate error
        actual = float(row[TARGET_COL])
        error = actual - final_pred
        
        # Online learning: Update SGD with residual
        residual_target = np.array([actual - xgb_pred], dtype=float)
        sgd_model.partial_fit(X_row_scaled, residual_target)
        
        # Resource allocation recommendations
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
    app.run(debug=True)
```

## 🔧 Feature Engineering Details

### Temporal Features
```python
feats["hour"] = df["timestamp"].dt.hour        # 0-23
feats["day"] = df["timestamp"].dt.day          # 1-31
feats["month"] = df["timestamp"].dt.month      # 1-12
```

### Cyclical Encoding
```python
# Converts linear hour (0-23) to circular representation
angle = 2.0 * np.pi * hour / 24
feats["hour_sin"] = np.sin(angle)
feats["hour_cos"] = np.cos(angle)
```

### Lag Features
Creates historical context by shifting values:
```python
for lag in [1, 2, 3, 6, 12, 24]:  # 1 to 24-step lags
    feats[f"cpu_utilization_lag_{lag}"] = df["cpu_utilization"].shift(lag)
```

### Rolling Statistics
```python
for window in [3, 6, 12, 24]:  # 3 to 24-step windows
    feats[f"cpu_utilization_roll_mean_{window}"] = df["cpu_utilization"].rolling(window).mean()
    feats[f"cpu_utilization_roll_std_{window}"] = df["cpu_utilization"].rolling(window).std()
    feats[f"cpu_utilization_roll_min_{window}"] = df["cpu_utilization"].rolling(window).min()
    feats[f"cpu_utilization_roll_max_{window}"] = df["cpu_utilization"].rolling(window).max()
```

### Interaction Features
```python
feats["cpu_mem_ratio"] = cpu_util / (memory_usage + 1e-6)
feats["cpu_workload"] = cpu_util * workload
feats["mem_workload"] = memory_usage * workload
```

## 📈 Model Configuration

### XGBoost
```python
XGBRegressor(
    n_estimators=600,           # 600 boosting rounds
    max_depth=6,                # Tree depth
    learning_rate=0.03,         # Shrinkage
    subsample=0.9,              # Row sampling
    colsample_bytree=0.9,       # Feature sampling
    reg_alpha=0.01,             # L1 regularization
    reg_lambda=1.0,             # L2 regularization
    random_state=42,
    objective="reg:squarederror",
    tree_method="hist",
)
```

### SGD (Online Learner for Residuals)
```python
SGDRegressor(
    loss="squared_error",
    penalty="l2",
    alpha=1e-4,                 # Regularization strength
    learning_rate="invscaling", # Adaptive learning rate
    eta0=0.01,                  # Initial learning rate
    power_t=0.25,               # Power for invscaling
    max_iter=2000,
    tol=1e-4,
    random_state=42,
)
```

## 🌐 API Endpoints

### GET `/`
Serves the web interface (index.html)

### GET `/predict_next`
Returns next sequential prediction

**Response:**
```json
{
  "done": false,
  "time": "2024-01-15 10:00:00",
  "predicted": 625.341234,
  "actual": 628.500000,
  "error": 3.158766,
  "predicted_servers": 63,
  "actual_servers": 63,
  "warm_servers": 4,
  "step": 1,
  "remaining": 5999
}
```

## 🎯 Ensemble Strategy

**Two-Stage Ensemble:**
1. **Primary Model (XGBoost)**: Main prediction from complex non-linear patterns
2. **Correction Layer (SGD)**: Learns residuals from XGBoost and adapts online

**Benefits:**
- Combines tree-based stability with linear residual adaptation
- Online learning (partial_fit) allows real-time model improvement
- Reduces overfitting through ensemble averaging

## 📊 Expected Performance

- **Training samples**: 20,000 rows
- **Test samples**: 6,000 rows (rows 20,001-26,000)
- **Feature count**: ~100+ engineered features
- **Online updates**: SGD retrains on each test prediction

## 🔍 Metrics Tracked

```python
{
    "predicted": float,         # Model prediction
    "actual": float,            # Ground truth
    "error": float,             # Prediction error
    "predicted_servers": int,   # Recommended servers
    "actual_servers": int,      # Actual servers needed
    "warm_servers": int,        # Pre-warmed standby servers
    "step": int,                # Current prediction step
    "remaining": int,           # Remaining test samples
}
```

## 📝 Requirements

```
Flask>=2.0.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
```

## 🚦 Running the System

```bash
# 1. Train models
python train.py --data dataset.csv

# 2. Start web server
python app.py

# 3. Open browser
# Go to http://localhost:5000

# 4. Click "Next" to see predictions sequentially
```

## 📚 Research References

See the `Research papers/` folder for:
- Cloud resource allocation methodologies
- Machine learning for infrastructure optimization
- AutoScaling and resource management strategies
- Related AI/ML papers on prediction systems

## 👤 Author

**Srivan Kumar**  
AI-Driven Cloud Resource Allocation System  
Final Year Project

## 📄 License

This project is part of an academic final year project.

---

**Last Updated**: May 2026
