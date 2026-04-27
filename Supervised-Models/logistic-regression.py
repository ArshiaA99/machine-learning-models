"""
A simple example of Logistic Regression using scikit-learn.
Demonstrates synthetic data generation, training, and evaluation.
"""
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

class LogisticRegressionModel:
    def make_classification_data(self):
        """Generate synthetic binary classification dataset."""
        X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
        return X, y

    def train_test_split(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
        """Split data into training and test sets."""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
        """Train logistic regression model."""
        model = LogisticRegression()
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, y_test: np.ndarray, y_pred: np.ndarray) -> dict:
        """Return evaluation metrics."""
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred)
        }

if __name__ == "__main__":
    model = LogisticRegressionModel()
    X, y = model.make_classification_data()
    X_train, X_test, y_train, y_test = model.train_test_split(X, y)
    logistic_model = model.train_model(X_train, y_train)
    y_pred = logistic_model.predict(X_test)

    results = model.evaluate_model(y_test, y_pred)
    print(f"Accuracy: {results['accuracy']}")
    print(f"Confusion Matrix:\n{results['confusion_matrix']}")
    print(f"Classification Report:\n{results['classification_report']}")
