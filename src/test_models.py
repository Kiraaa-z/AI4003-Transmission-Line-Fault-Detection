
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from logistic import predict
from metrics import accuracy, confusion_matrix


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"

FEATURES = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]


def evaluate(name, y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)

    print(f"\n{name}")
    print(f"Test accuracy: {accuracy(y_true, y_pred):.4f}")
    print("Confusion matrix:")
    print(matrix)
    print(f"Missed faults: {int(matrix[1, 0])}")


def main():
    # Load the test split
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    X_test = test_df[FEATURES].to_numpy(dtype=float)
    y_test = test_df["fault"].to_numpy(dtype=int)

    # Load Set R coefficients
    with open(
        RESULTS_DIR / "logistic_set_r.json",
        encoding="utf-8",
    ) as file:
        set_r = json.load(file)

    theta_r = np.asarray(set_r["theta"], dtype=float)
    X_test_r = np.column_stack([np.ones(len(X_test)), X_test])

    evaluate(
        "Logistic Regression - Set R",
        y_test,
        predict(X_test_r, theta_r),
    )

    # Build Set Q from raw features and training-only scaling
    train_raw = pd.read_csv(DATA_DIR / "train_raw.csv")
    test_raw = pd.read_csv(DATA_DIR / "test_raw.csv")

    train_features = train_raw[FEATURES].to_numpy(dtype=float)
    test_features = test_raw[FEATURES].to_numpy(dtype=float)

    q_train = np.column_stack(
        [train_features, train_features ** 2]
    )
    q_test = np.column_stack(
        [test_features, test_features ** 2]
    )

    q_mean = q_train.mean(axis=0)
    q_std = q_train.std(axis=0, ddof=1)
    X_test_q = (q_test - q_mean) / q_std

    with open(
        RESULTS_DIR / "logistic_set_q.json",
        encoding="utf-8",
    ) as file:
        set_q = json.load(file)

    theta_q = np.asarray(set_q["theta"], dtype=float)
    X_test_q = np.column_stack(
        [np.ones(len(X_test_q)), X_test_q]
    )

    evaluate(
        "Logistic Regression - Set Q",
        y_test,
        predict(X_test_q, theta_q),
    )

    # Load the selected SVM models
    linear_svm = joblib.load(
        MODELS_DIR / "best_linear_svm.joblib"
    )
    rbf_svm = joblib.load(
        MODELS_DIR / "best_rbf_svm.joblib"
    )

    evaluate(
        "Best Linear SVM",
        y_test,
        linear_svm.predict(X_test),
    )

    evaluate(
        "Best RBF SVM",
        y_test,
        rbf_svm.predict(X_test),
    )


if __name__ == "__main__":
    main()
