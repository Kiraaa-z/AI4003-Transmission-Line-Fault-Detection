
import numpy as np


def sigmoid(z):
    """Convert a score into a probability between 0 and 1."""
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def hypothesis(X, theta):
    """Calculate the predicted probability for each row."""
    return sigmoid(X @ theta)


def cost(X, y, theta):
    """Calculate logistic regression cross-entropy cost."""
    probabilities = hypothesis(X, theta)

    # Avoid taking the logarithm of zero
    probabilities = np.clip(probabilities, 1e-12, 1 - 1e-12)

    total = y * np.log(probabilities)
    total += (1 - y) * np.log(1 - probabilities)

    return -np.mean(total)


def fit_logistic(X, y, alpha, n_iters):
    """Train logistic regression using batch gradient descent."""
    rows, columns = X.shape
    theta = np.zeros(columns)
    costs = []

    for _ in range(n_iters):
        probabilities = hypothesis(X, theta)
        gradient = X.T @ (y - probabilities) / rows
        theta = theta + alpha * gradient

        costs.append(cost(X, y, theta))

    return theta, costs


def predict(X, theta):
    """Predict 1 for a fault and 0 for no fault."""
    probabilities = hypothesis(X, theta)
    return (probabilities >= 0.5).astype(int)
