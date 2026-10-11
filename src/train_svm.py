
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from metrics import accuracy


# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


# Use only the six standardised Set R features
FEATURES = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]


def load_split(filename):
    """Load one prepared dataset split."""
    df = pd.read_csv(DATA_DIR / filename)
    X = df[FEATURES].to_numpy(dtype=float)
    y = df["fault"].to_numpy(dtype=int)
    return X, y


def main():
    X_train, y_train = load_split("train.csv")
    X_val, y_val = load_split("validation.csv")

    results = []
    best_linear = None
    best_rbf = None

    # Test linear SVM with the required C values
    print("\nLINEAR SVM")
    for C in [0.1, 1, 10, 100]:
        model = SVC(kernel="linear", C=C)
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        val_accuracy = accuracy(y_val, predictions)
        support_vectors = int(model.n_support_.sum())

        results.append({
            "Kernel": "linear",
            "C": C,
            "Gamma": "N/A",
            "Validation Accuracy": val_accuracy,
            "Support Vectors": support_vectors,
        })

        print(
            f"C={C:<5} | Validation accuracy={val_accuracy:.4f} "
            f"| Support vectors={support_vectors}"
        )

        if (
            best_linear is None
            or val_accuracy > best_linear["accuracy"]
        ):
            best_linear = {
                "model": model,
                "C": C,
                "accuracy": val_accuracy,
            }

    # Test RBF SVM with the required C and gamma values
    print("\nRBF SVM")
    for C in [0.1, 1, 10, 100]:
        for gamma in [0.1, 1]:
            model = SVC(kernel="rbf", C=C, gamma=gamma)
            model.fit(X_train, y_train)

            predictions = model.predict(X_val)
            val_accuracy = accuracy(y_val, predictions)
            support_vectors = int(model.n_support_.sum())

            results.append({
                "Kernel": "rbf",
                "C": C,
                "Gamma": gamma,
                "Validation Accuracy": val_accuracy,
                "Support Vectors": support_vectors,
            })

            print(
                f"C={C:<5} | gamma={gamma:<3} "
                f"| Validation accuracy={val_accuracy:.4f} "
                f"| Support vectors={support_vectors}"
            )

            if (
                best_rbf is None
                or val_accuracy > best_rbf["accuracy"]
            ):
                best_rbf = {
                    "model": model,
                    "C": C,
                    "gamma": gamma,
                    "accuracy": val_accuracy,
                }

    # Save every configuration's validation results
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        RESULTS_DIR / "svm_results.csv",
        index=False,
    )
    
    # Plot validation accuracy against C
    plt.figure(figsize=(8, 5))

    # Linear SVM
    linear_results = results_df[results_df["Kernel"] == "linear"]
    plt.plot(
        linear_results["C"],
        linear_results["Validation Accuracy"],
        marker="o",
        label="Linear SVM",
    )

    # RBF SVM for each gamma value
    rbf_results = results_df[results_df["Kernel"] == "rbf"]

    for gamma in [0.1, 1]:
        gamma_results = rbf_results[rbf_results["Gamma"] == gamma]
        plt.plot(
            gamma_results["C"],
            gamma_results["Validation Accuracy"],
            marker="o",
            label=f"RBF SVM (gamma={gamma})",
        )

    plt.xscale("log")
    plt.xlabel("C (log scale)")
    plt.ylabel("Validation Accuracy")
    plt.title("SVM Validation Accuracy vs C")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(RESULTS_DIR / "svm_validation_accuracy.png", dpi=150)
    plt.close()

    print("results/svm_validation_accuracy.png")


    # Save the best model for each kernel
    joblib.dump(
        best_linear["model"],
        MODELS_DIR / "best_linear_svm.joblib",
    )
    joblib.dump(
        best_rbf["model"],
        MODELS_DIR / "best_rbf_svm.joblib",
    )

    print("\nBEST LINEAR SVM")
    print(f"C: {best_linear['C']}")
    print(f"Validation accuracy: {best_linear['accuracy']:.4f}")

    print("\nBEST RBF SVM")
    print(f"C: {best_rbf['C']}")
    print(f"Gamma: {best_rbf['gamma']}")
    print(f"Validation accuracy: {best_rbf['accuracy']:.4f}")

    print("\nSaved:")
    print("results/svm_results.csv")
    print("results/models/best_linear_svm.joblib")
    print("results/models/best_rbf_svm.joblib")


if __name__ == "__main__":
    main()
