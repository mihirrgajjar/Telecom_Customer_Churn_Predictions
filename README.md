# Customer Churn Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit Learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Modeling-006400?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)

A practical machine learning project for predicting customer churn from telecom customer behavior. The project moves from notebook-based exploration to trained model artifacts and a Streamlit dashboard where users can explore churn patterns, compare models, and make single or batch predictions.

The goal is simple: help a business identify customers who are likely to leave before they actually do.

## Table Of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Architecture](#project-architecture)
- [Machine Learning Workflow](#machine-learning-workflow)
- [Model Performance](#model-performance)
- [Getting Started](#getting-started)
- [Running The Dashboard](#running-the-dashboard)
- [Using Batch Prediction](#using-batch-prediction)
- [Repository Notes](#repository-notes)
- [Future Improvements](#future-improvements)

## Project Overview

Customer churn is one of the most important signals for subscription and service businesses. This project uses customer profile, billing, contract, and service usage data to estimate whether a customer is likely to churn.

The dataset contains 7,043 customer records and 21 original columns, including:

- Customer demographics such as gender, senior citizen status, partner, and dependents.
- Account information such as tenure, contract type, billing method, and payment method.
- Service usage fields such as internet service, online security, tech support, streaming TV, and streaming movies.
- Billing values such as monthly charges and total charges.
- Target label: `Churn`.

## Key Features

- Exploratory data analysis for churn patterns and customer behavior.
- Data cleaning for missing and inconsistent values.
- Feature encoding and scaling for machine learning models.
- SMOTE-based class balancing to reduce churn class imbalance.
- Multiple trained models for comparison.
- Saved model artifacts for reusable predictions.
- Streamlit dashboard for interactive exploration.
- Single customer churn prediction form.
- Batch CSV upload for multiple churn predictions.
- Downloadable batch prediction results.

## Tech Stack

| Area | Tools |
| --- | --- |
| Language | Python |
| Data handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine learning | scikit-learn, imbalanced-learn, XGBoost |
| Model storage | Joblib |
| App interface | Streamlit |
| Development | Jupyter Notebook |

## Project Architecture

```text
customer_churn_prediction/
|
|-- app/
|   `-- dashboard.py              # Streamlit dashboard and prediction UI
|
|-- models/
|   |-- best_model.pkl            # Best selected model used by the app
|   |-- dt_model.pkl              # Decision Tree model
|   |-- feature_columns.pkl       # Final training feature order
|   |-- lr_model.pkl              # Logistic Regression model
|   |-- rf_model.pkl              # Random Forest model
|   |-- scaler.pkl                # Fitted scaler for numeric columns
|   `-- xgb_model.pkl             # XGBoost model
|
|-- churn_analysis.ipynb          # Full analysis, preprocessing, training, and saving flow
|-- CustomerChurn.csv             # Source customer churn dataset
|-- requirements.txt              # Project dependencies
`-- README.md                     # Project documentation
```

## Machine Learning Workflow

The notebook follows a clear training pipeline:

1. Load and understand the customer churn dataset.
2. Clean the data, including `TotalCharges` conversion and missing value handling.
3. Explore churn behavior across contracts, tenure, charges, and service types.
4. Preprocess features with one-hot encoding and numeric scaling.
5. Handle class imbalance with SMOTE.
6. Train baseline and tree-based classifiers.
7. Evaluate models with accuracy, precision, recall, and F1 score.
8. Save the best model, supporting models, scaler, and feature columns.
9. Reuse saved artifacts inside the Streamlit dashboard.

## Model Performance

The notebook compares four models. Since churn detection often cares more about catching at-risk customers than maximizing raw accuracy, recall is an important metric.

| Model | Accuracy | Precision | Recall | F1 Score |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 73.10 | 49.53 | 69.79 | 57.94 |
| Decision Tree | 73.74 | 50.39 | 68.72 | 58.14 |
| Random Forest | 74.95 | 52.38 | 61.76 | 56.69 |
| XGBoost | 75.80 | 53.37 | 69.79 | 60.49 |

The saved `best_model.pkl` points to the XGBoost model in the current project state.

## Getting Started

Clone or open the project locally, then create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The dashboard imports Plotly. If Plotly is not already installed in your environment, install it as well:

```bash
pip install plotly
```

## Running The Dashboard

Run the Streamlit app from the project root:

```bash
streamlit run app/dashboard.py
```

The dashboard includes five main views:

- `Overview`: high-level customer count, churn rate, and billing metrics.
- `Churn Analysis`: visual breakdowns by tenure, charges, internet service, and payment method.
- `Model Performance`: model comparison with confusion matrices and metric tables.
- `Predict Single Customer`: form-based prediction for one customer.
- `Batch Predictions`: CSV upload for bulk churn scoring.

## Using Batch Prediction

For batch prediction, upload a CSV with the same customer feature columns used in the training data. The app will:

- Clean and preprocess the uploaded records.
- Align columns with the saved training feature order.
- Apply the fitted scaler to numeric values.
- Predict churn for each customer.
- Add churn probability as a percentage.
- Let you download the prediction results as a CSV.

## Repository Notes

- Keep `CustomerChurn.csv` in the project root because both the notebook and dashboard expect it there.
- Keep the `models/` directory in place because the dashboard loads model artifacts from that path.
- The app expects the saved scaler and feature column list to match the model training pipeline.
- If you retrain the model, regenerate all files in `models/` together to avoid feature mismatch errors.

## Future Improvements

- Add Plotly to `requirements.txt` so dashboard setup is fully reproducible.
- Add a small sample CSV for batch prediction testing.
- Add model versioning for cleaner retraining and deployment.
- Add automated tests for preprocessing consistency.
- Add probability threshold controls in the dashboard.
- Add explainability with feature importance or SHAP values.

## License

No license file is currently included. Add one before using this project in a public or production setting.
