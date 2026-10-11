
import numpy as np


def accuracy(y_true, y_pred):
    """Calculate the fraction of correct predictions."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)


def confusion_matrix(y_true, y_pred):
    """Return counts for true negatives, false alarms, missed faults and true positives."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_negative = np.sum((y_true == 0) & (y_pred == 0))
    false_alarm = np.sum((y_true == 0) & (y_pred == 1))
    missed_fault = np.sum((y_true == 1) & (y_pred == 0))
    true_positive = np.sum((y_true == 1) & (y_pred == 1))

    return np.array([
        [true_negative, false_alarm],
        [missed_fault, true_positive]
    ])
