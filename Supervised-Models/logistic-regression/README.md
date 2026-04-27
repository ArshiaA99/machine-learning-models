# 📘 Logistic Regression Model Example

## 🧠 What is Logistic Regression?
**Logistic Regression** is a supervised machine learning algorithm used for **classification tasks** — predicting *categorical outcomes* such as yes/no, spam/not spam, or disease/no disease.  
It models the probability that a given input belongs to a certain class using a **logistic function** (also known as the sigmoid function), which outputs values between 0 and 1.

Common real‑world uses include:
- Email spam detection  
- Customer churn prediction  
- Credit risk assessment  
- Medical diagnosis (e.g., predicting disease presence)

---

## 🎯 Project Overview
This project demonstrates a minimal, clean implementation of a logistic regression model using **scikit‑learn**.  
It walks through the full training pipeline — from synthetic dataset generation to model training, evaluation, and reporting of performance metrics.

### Key features:
- Uses `make_classification()` to synthesize labeled data  
- Splits data into training and testing sets  
- Trains a `LogisticRegression` model  
- Outputs **Accuracy**, **Confusion Matrix**, and a detailed **Classification Report**

### How It Works?
The script will automatically:
1. Generate a synthetic binary classification dataset
2. Split the data into training and test sets
3. Train a logistic regression model
4. Evaluate the model and print the results to the console

### 📊 Example Output
```text
Accuracy: 0.83
Confusion Matrix:
[[75 14]
 [20 91]]
Classification Report:
precision    recall  f1-score   support

0       0.79      0.84      0.82        89
1       0.87      0.82      0.84       111

accuracy                           0.83       200
   macro avg       0.83      0.83      0.83       200
weighted avg       0.83      0.83      0.83       200
```
---

### Credits: Created by Arshia K ([Github: ArshiaA99](https://github.com/ArshiaA99))
