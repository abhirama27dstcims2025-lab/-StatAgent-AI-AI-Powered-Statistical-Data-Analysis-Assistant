import pandas as pd
from scipy import stats


# ==========================================================
# 1. SHAPIRO-WILK NORMALITY TEST
# ==========================================================

def shapiro_test(df, column):

    data = df[column].dropna()

    if len(data) < 3:
        return "At least 3 observations are required."

    if len(data) > 5000:
        data = data.sample(5000, random_state=42)

    statistic, p_value = stats.shapiro(data)

    return {
        "Test": "Shapiro-Wilk Test",
        "Statistic": statistic,
        "P-value": p_value,
        "Conclusion": (
            "Data appears normally distributed"
            if p_value > 0.05
            else "Evidence against normality"
        )
    }


# ==========================================================
# 2. CHI-SQUARE TEST
# ==========================================================

def chi_square_test(df, column1, column2):

    contingency_table = pd.crosstab(
        df[column1],
        df[column2]
    )

    if contingency_table.empty:
        return "Insufficient data for Chi-Square test."

    statistic, p_value, dof, expected = (
        stats.chi2_contingency(contingency_table)
    )

    return {
        "Test": "Chi-Square Test",
        "Statistic": statistic,
        "P-value": p_value,
        "Degrees of Freedom": dof,
        "Conclusion": (
            "Significant association exists"
            if p_value < 0.05
            else "No significant association found"
        )
    }


# ==========================================================
# 3. RECOMMEND STATISTICAL TESTS
# ==========================================================

def recommend_statistical_tests(
    df, target, problem, numeric_cols, categorical_cols
):

    recommendations = []

    if target is None:

        return [
            (
                "Exploratory Data Analysis",
                "Start with descriptive statistics and visualizations."
            ),
            (
                "Cluster Analysis",
                "Explore groups in the data without a target variable."
            )
        ]

    if problem == "Binary Classification":

        if categorical_cols:
            recommendations.append((
                "Chi-Square Test",
                "Examine associations between categorical variables."
            ))

        if numeric_cols:
            recommendations.append((
                "Independent t-Test",
                "Compare numeric means between two target groups."
            ))

        if len(numeric_cols) > 1:
            recommendations.append((
                "Pearson Correlation",
                "Examine linear relationships between numeric variables."
            ))

    elif problem == "Multi-class Classification":

        if numeric_cols:
            recommendations.append((
                "One-Way ANOVA",
                "Compare numeric means across target groups."
            ))

        if categorical_cols:
            recommendations.append((
                "Chi-Square Test",
                "Examine associations between categorical variables."
            ))

    elif problem == "Regression":

        recommendations.extend([
            (
                "Pearson Correlation",
                "Measure linear relationships between numeric variables."
            ),
            (
                "Multiple Linear Regression",
                "Model a numeric outcome using multiple predictors."
            ),
            (
                "Variance Inflation Factor (VIF)",
                "Check for multicollinearity among predictors."
            )
        ])

    elif problem == "Time Series Forecasting":

        recommendations.extend([
            (
                "ADF Test",
                "Check whether the time series is stationary."
            ),
            (
                "ACF/PACF",
                "Examine autocorrelation and temporal dependencies."
            )
        ])

    else:

        recommendations.append((
            "Cluster Analysis",
            "Explore groups in the data without a target variable."
        ))

    return recommendations