import pandas as pd


# ==========================================================
# 1. DETECT MACHINE LEARNING PROBLEM TYPE
# ==========================================================

def detect_problem_type(df, target):

    if target is None:
        return "Clustering"

    if target not in df.columns:
        return "Invalid Target"

    target_data = df[target].dropna()

    if target_data.empty:
        return "Invalid Target"

    if pd.api.types.is_datetime64_any_dtype(target_data):
        return "Time Series Forecasting"

    if pd.api.types.is_numeric_dtype(target_data):

        unique_values = target_data.nunique()

        if unique_values == 2:
            return "Binary Classification"

        elif unique_values <= 10:
            return "Multi-class Classification"

        else:
            return "Regression"

    else:

        unique_values = target_data.nunique()

        if unique_values == 2:
            return "Binary Classification"

        else:
            return "Multi-class Classification"


# ==========================================================
# 2. RECOMMEND MACHINE LEARNING MODELS
# ==========================================================

def recommend_ml_models(problem):

    recommendations = []

    if problem == "Binary Classification":

        recommendations = [
            ("Logistic Regression", "Baseline classification model."),
            ("Decision Tree", "Learns decision rules from data."),
            ("Random Forest", "Combines multiple decision trees."),
            ("XGBoost", "Gradient boosting for classification.")
        ]

    elif problem == "Multi-class Classification":

        recommendations = [
            ("Logistic Regression", "Supports multi-class classification."),
            ("Decision Tree", "Classifies observations using decision rules."),
            ("Random Forest", "Ensemble learning for multiple classes."),
            ("XGBoost", "Boosting-based classification model.")
        ]

    elif problem == "Regression":

        recommendations = [
            ("Linear Regression", "Models linear relationships."),
            ("Decision Tree Regressor", "Captures nonlinear relationships."),
            ("Random Forest Regressor", "Ensemble-based regression."),
            ("XGBoost Regressor", "Boosting-based regression model.")
        ]

    elif problem == "Clustering":

        recommendations = [
            ("K-Means", "Groups observations into clusters."),
            ("Hierarchical Clustering", "Builds hierarchical groups."),
            ("DBSCAN", "Finds density-based clusters.")
        ]

    elif problem == "Time Series Forecasting":

        recommendations = [
            ("ARIMA", "Models time-dependent patterns."),
            ("SARIMA", "Handles seasonality in time series."),
            ("Prophet", "Forecasts trends and seasonal patterns.")
        ]

    return recommendations