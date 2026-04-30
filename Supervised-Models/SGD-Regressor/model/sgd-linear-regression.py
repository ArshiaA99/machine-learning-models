import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score

class LinearRegressionModel:
    def __init__(self):
        pass
    
    def load_data(self, file_path):
        """Load the data from file_path."""
        data = pd.read_csv(file_path)
        return data
    
    def drop_feature(self, data, feature_name):
        """Drop unnecessary columns."""
        data = data.drop(columns=[feature_name])
        return data
    
    def create_feature_label(self, data, feature_name):
        """Create features and the target."""
        X = data.drop(columns=[feature_name])
        y = data[feature_name]
        return X, y
    
    def train_test_split(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and test sets."""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def standardize_data(self):
        """Standardize the data using StandardScaler() and OneHotEncoder()."""
        numeric_features = ['SquareFeet', 'Bedrooms', 'Bathrooms']
        categorical_features = ['Neighborhood']

        preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(), categorical_features)
                ]
        )

        return preprocessor
    
    def train_model(self, X_train, y_train, preprocessor):
            """Train linear regression (SGD) model."""
            pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('model', SGDRegressor(loss="squared_error", max_iter=1000, learning_rate="invscaling", eta0=0.01))])
            pipeline.fit(X_train, y_train)
            return pipeline

    def evaluate_model(self, y_test, y_pred) -> dict:
        """Return evaluation metrics for regression."""
        return {
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": root_mean_squared_error(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred)
        }

if __name__ == "__main__":
    model = LinearRegressionModel()
    data = model.load_data("data/housing_price_dataset.csv")
    data = model.drop_feature(data, "YearBuilt")
    X, y = model.create_feature_label(data, "Price")
    X_train, X_test, y_train, y_test = model.train_test_split(X, y)
    preprocessor = model.standardize_data()
    sgd_model = model.train_model(X_train, y_train, preprocessor)
    y_pred = sgd_model.predict(X_test)

    metrics = model.evaluate_model(y_test, y_pred)
    print(metrics)
