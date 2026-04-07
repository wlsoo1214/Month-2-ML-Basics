# src/evaluator.py
# Reusable evaluation harness
# Generates full report for any model 

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)
import numpy as np

class MLEvaluator:
    # decorator dont need to interact with the class instance
    # utility function that do job w/o caring about object's state
    @staticmethod
    def evaluate_classification(y_true, y_pred, prefix="Model"):
        """Metrics for 'Is High Value' prediction """
        print(f"\n-- {prefix} Classification Report ---")
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        print(f"Precision: {precision_score(y_true, y_pred):.4f}")
        print(f"Recall: {recall_score(y_true, y_pred):.4f}")
        print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
        print("-" * 40)

    @staticmethod
    def evaluate_regression(y_true, y_pred, prefix="Model"):
        """Metrics for 'Price' prediction """
        print(f"\n-- {prefix} Regression Report ---")
        print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")
        print(f"MSE: {mean_squared_error(y_true, y_pred):.4f}")
        print(f"R2 Score: {r2_score(y_true, y_pred):.4f}")
        print("-" * 40)

if __name__ == "__main__":
    # Mock data to test the harness
    test_true = [1, 0, 1, 1, 0]
    test_pred = [1, 0, 0, 1, 0]

    # use the defined function under the class MLEvaluator
    MLEvaluator.evaluate_classification(test_true, test_pred, "Test Run")