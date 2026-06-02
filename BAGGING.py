import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==============================================================================
# LOAD DATA
# ==============================================================================

csv_filename = "GP_training_data.csv"

print(f"Loading dataset: {csv_filename}")

df = pd.read_csv(csv_filename)

feature_cols = [
    'Thickness_mm',
    'YoungsModulus_MPa',
    'Density_t_mm3',
    'PoissonsRatio'
]

target_cols = [
    'Freq1_Hz',
    'Freq2_Hz',
    'Freq3_Hz'
]

X = df[feature_cols].values
y = df[target_cols].values

# ==============================================================================
# TRAIN TEST SPLIT
# ==============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==============================================================================
# SCALE FEATURES
# ==============================================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================================================================
# PLOT SETUP
# ==============================================================================

fig, axes = plt.subplots(
    3,
    2,
    figsize=(12, 15)
)

results = []

print("\n=============== BAGGING RESULTS ===============\n")

# ==============================================================================
# TRAIN MODELS
# ==============================================================================

for i, target in enumerate(target_cols):

    model = BaggingRegressor(
        estimator=DecisionTreeRegressor(),
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train_scaled,
        y_train[:, i]
    )

    y_pred = model.predict(
        X_test_scaled
    )

    mae = mean_absolute_error(
        y_test[:, i],
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test[:, i],
            y_pred
        )
    )

    r2 = r2_score(
        y_test[:, i],
        y_pred
    )

    mape = np.mean(
        np.abs(
            (y_test[:, i] - y_pred)
            / y_test[:, i]
        )
    ) * 100

    print(f"{target}")
    print(f"MAE  = {mae:.6f}")
    print(f"RMSE = {rmse:.6f}")
    print(f"R²   = {r2:.6f}")
    print(f"MAPE = {mape:.4f}%")
    print("-" * 40)

    results.append([
        target,
        mae,
        rmse,
        r2,
        mape
    ])

    # ==========================================================
    # Predicted vs Actual
    # ==========================================================

    ax1 = axes[i, 0]

    ax1.scatter(
        y_test[:, i],
        y_pred,
        color='darkgreen',
        edgecolors='black',
        s=60
    )

    min_val = min(
        y_test[:, i].min(),
        y_pred.min()
    )

    max_val = max(
        y_test[:, i].max(),
        y_pred.max()
    )

    ax1.plot(
        [min_val, max_val],
        [min_val, max_val],
        'r--',
        linewidth=2
    )

    ax1.set_title(
        f"{target} | R²={r2:.4f}"
    )

    ax1.set_xlabel("Actual Frequency (Hz)")
    ax1.set_ylabel("Predicted Frequency (Hz)")
    ax1.grid(True, linestyle=":")

    # ==========================================================
    # Residual Plot
    # ==========================================================

    residuals = y_test[:, i] - y_pred

    ax2 = axes[i, 1]

    ax2.scatter(
        y_pred,
        residuals,
        color='purple',
        edgecolors='black',
        s=60
    )

    ax2.axhline(
        0,
        color='black',
        linewidth=1.5
    )

    ax2.set_title(
        f"{target} Residuals"
    )

    ax2.set_xlabel("Predicted Frequency (Hz)")
    ax2.set_ylabel("Residual Error (Hz)")
    ax2.grid(True, linestyle=":")

# ==============================================================================
# SAVE OUTPUTS
# ==============================================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Target",
        "MAE",
        "RMSE",
        "R2",
        "MAPE (%)"
    ]
)

results_df.to_csv(
    "Bagging_Metrics.csv",
    index=False
)

plt.tight_layout()

plt.savefig(
    "Bagging_Evaluation.png",
    dpi=300
)

plt.close()

print("\nSaved:")
print("Bagging_Evaluation.png")
print("Bagging_Metrics.csv")