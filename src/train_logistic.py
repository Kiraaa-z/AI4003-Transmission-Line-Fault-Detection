
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from logistic import fit_logistic, predict, cost
from metrics import accuracy


# Find the project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Load the prepared datasets
train_raw = pd.read_csv(DATA_DIR / "train_raw.csv")
val_raw = pd.read_csv(DATA_DIR / "validation_raw.csv")

train_std = pd.read_csv(DATA_DIR / "train.csv")
val_std = pd.read_csv(DATA_DIR / "validation.csv")

feature_columns = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]
label_column = "fault"

# Prepare the target labels
y_train = train_raw[label_column].to_numpy(dtype=int)
y_val = val_raw[label_column].to_numpy(dtype=int)

# Set R: six standardised original measurements
X_train_r = train_std[feature_columns].to_numpy(dtype=float)
X_val_r = val_std[feature_columns].to_numpy(dtype=float)

# Set Q: original measurements plus their squares
X_train_q_raw = train_raw[feature_columns].to_numpy(dtype=float)
X_val_q_raw = val_raw[feature_columns].to_numpy(dtype=float)

X_train_q_raw = np.column_stack(
    [X_train_q_raw, X_train_q_raw ** 2]
)
X_val_q_raw = np.column_stack(
    [X_val_q_raw, X_val_q_raw ** 2]
)

# Standardise all 12 Set Q features using training statistics only
q_mean = X_train_q_raw.mean(axis=0)
q_std = X_train_q_raw.std(axis=0, ddof=1)
q_std[q_std == 0] = 1

X_train_q = (X_train_q_raw - q_mean) / q_std
X_val_q = (X_val_q_raw - q_mean) / q_std

# Add an intercept column to both feature sets
X_train_r = np.column_stack(
    [np.ones(len(X_train_r)), X_train_r]
)
X_val_r = np.column_stack(
    [np.ones(len(X_val_r)), X_val_r]
)

X_train_q = np.column_stack(
    [np.ones(len(X_train_q)), X_train_q]
)
X_val_q = np.column_stack(
    [np.ones(len(X_val_q)), X_val_q]
)


def tune_feature_set(name, X_train, X_val):
    """Test learning rates and train the final model."""

    learning_rates = [0.01, 0.1, 1, 10]
    tuning_results = {}
    best_alpha = None

    print(f"\nTuning {name}")

    # Try every required learning rate
    for alpha in learning_rates:
        theta, costs = fit_logistic(
            X_train, y_train, alpha, 2000
        )

        costs = np.array(costs)

        # Check whether cost decreases at every iteration
        decreases_every_time = np.all(
            np.diff(costs) < 0
        )

        tuning_results[alpha] = costs

        print(
            f"alpha={alpha}: "
            f"final cost={costs[-1]:.6f}, "
            f"decreases every iteration={decreases_every_time}"
        )

        if decreases_every_time:
            best_alpha = alpha

    if best_alpha is None:
        raise RuntimeError(
            f"No learning rate produced a strictly decreasing cost for {name}."
        )

    # Plot the cost curves for all learning rates
    plt.figure(figsize=(9, 5))

    for alpha, costs in tuning_results.items():
        plt.plot(costs, label=f"alpha={alpha}")

    plt.title(f"Logistic Regression Cost Curves - {name}")
    plt.xlabel("Iteration")
    plt.ylabel("Cost J(theta)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_name = f"cost_curves_{name.lower().replace(' ', '_')}.png"
    plt.savefig(RESULTS_DIR / plot_name, dpi=300)
    plt.close()

    # Train the final model using the selected learning rate
    theta, final_costs = fit_logistic(
        X_train, y_train, best_alpha, 2000
    )

    train_predictions = predict(X_train, theta)
    val_predictions = predict(X_val, theta)

    train_accuracy = accuracy(y_train, train_predictions)
    val_accuracy = accuracy(y_val, val_predictions)

    print(f"\n{name} selected alpha: {best_alpha}")
    print(f"{name} final cost: {final_costs[-1]:.6f}")
    print(f"{name} training accuracy: {train_accuracy:.4f}")
    print(f"{name} validation accuracy: {val_accuracy:.4f}")
    print(f"{name} theta:")
    print(theta)

    return {
        "alpha": best_alpha,
        "theta": theta.tolist(),
        "final_cost": float(final_costs[-1]),
        "training_accuracy": float(train_accuracy),
        "validation_accuracy": float(val_accuracy),
        "costs": [float(value) for value in final_costs],
    }


# Train both required feature sets
results_r = tune_feature_set(
    "Set R", X_train_r, X_val_r
)
results_q = tune_feature_set(
    "Set Q", X_train_q, X_val_q
)

# Save Set R model parameters
model_parameters_r = {
    "feature_columns": ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"],
    "theta": results_r["theta"],
    "alpha": results_r["alpha"],
}

with open(
    RESULTS_DIR / "logistic_set_r.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(model_parameters_r, file, indent=4)

# Save the Set Q scaling information for later prediction
model_parameters = {
    "feature_columns": feature_columns,
    "q_mean": q_mean.tolist(),
    "q_std": q_std.tolist(),
    "theta": results_q["theta"],
    "alpha": results_q["alpha"],
}

with open(
    RESULTS_DIR / "logistic_set_q.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(model_parameters, file, indent=4)

# Save a results table
results_table = pd.DataFrame([
    {
        "Feature Set": "Set R",
        "Chosen Alpha": results_r["alpha"],
        "Final Cost": results_r["final_cost"],
        "Training Accuracy": results_r["training_accuracy"],
        "Validation Accuracy": results_r["validation_accuracy"],
    },
    {
        "Feature Set": "Set Q",
        "Chosen Alpha": results_q["alpha"],
        "Final Cost": results_q["final_cost"],
        "Training Accuracy": results_q["training_accuracy"],
        "Validation Accuracy": results_q["validation_accuracy"],
    },
])

results_table.to_csv(
    RESULTS_DIR / "logistic_results.csv",
    index=False
)

print("\nSaved logistic regression results.")
print(results_table.to_string(index=False))
