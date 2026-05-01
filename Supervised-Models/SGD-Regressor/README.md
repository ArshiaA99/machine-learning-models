# 📘 Linear Regression Model

## 🧠 What is Linear Regression?
Linear Regression is a fundamental supervised learning algorithm used for predicting continuous outcomes based on input features.

It models the linear relationship between independent variables (features) and a dependent variable (target) — assuming the change in output is proportional to the change in inputs.

Common real‑world applications include:

* Predicting housing prices
* Estimating sales revenue
* Forecasting energy consumption
* Determining employee salaries

## 🎯 Project Overview
This project demonstrates a clean, end‑to‑end implementation of a Linear Regression model using Stochastic Gradient Descent (SGD) via scikit‑learn.

The pipeline covers every step — from data loading to feature preprocessing, training, and evaluation — providing a transparent view of a modern machine learning workflow.

Key Features:
* Loads data from a CSV file (housing_price_dataset.csv)
* Removes unnecessary features (e.g., YearBuilt)
* Splits dataset into training and testing subsets
* Standardizes numerical columns and one‑hot encodes categorical ones
* Trains a SGDRegressor with the squared error loss
* Evaluates model accuracy using common regression metrics:
  * Mean Squared Error (MSE)
  *  Root Mean Squared Error (RMSE)
  *  Mean Absolute Error (MAE)
  *  R² (Coefficient of Determination)

## 💡 Technologies Used:
- Python 3.x
- NumPy
- Pandas
- scikit-learn


## ⚙️ How It Works
When you run the script, it will automatically:
1. Load the dataset from data/housing_price_dataset.csv
2. Drop the YearBuilt column (as an example)
3. Separate the data into features (X) and label (Price)
4. Split the dataset into training and testing sets
5. Preprocess data with scaling and encoding
6. Train a Linear Regression (SGD) model
7. Predict housing prices on the test set
8. Display performance metrics in the console

## 📊 Example Output
```text
{'mse': 2441180910.4864593, 'rmse': 49408.308111960876, 'mae': 39459.57495430301, 'r2': 0.5747037028827631}
```

## 🧾 Example Use Case
This project can serve as a lightweight, customizable template for regression modeling — ideal for:
* Real estate price prediction
* Predictive maintenance cost estimation
* Business forecasting models

## ⚙️ Potential Improvements
It is just a basic implimentation of this model to show how it is used, but it can be improved by:
* Adding more relevant features (e.g. GarageSize, LotArea, YearBuilt, distance to downtown, etc.)
* Trying different models — e.g. RandomForestRegressor, GradientBoostingRegressor
* Feature engineering:
* Log-transforming prices (np.log(Price))
* Creating interaction terms (Bedroom × Bathroom)
* Hyperparameter tuning for your SGDRegressor (adjust eta0, alpha, max_iter, penalty='l2')
* Checking outliers — overly large/small price entries can distort results.

### Credits: Created by Arshia K ([Github: ArshiaA99](https://github.com/ArshiaA99))
