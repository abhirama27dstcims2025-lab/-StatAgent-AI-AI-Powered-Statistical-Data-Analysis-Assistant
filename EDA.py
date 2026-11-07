import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# ==========================================================
# 1. UNIVARIATE ANALYSIS - NUMERICAL DISTRIBUTION
# ==========================================================

def plot_distribution(df, column):

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=df,
        x=column,
        kde=True,
        ax=ax
    )

    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")

    return fig


# ==========================================================
# 2. UNIVARIATE ANALYSIS - BOXPLOT
# ==========================================================

def plot_boxplot(df, column):

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.boxplot(
        data=df,
        y=column,
        ax=ax
    )

    ax.set_title(f"Boxplot of {column}")

    return fig


# ==========================================================
# 3. UNIVARIATE ANALYSIS - CATEGORICAL DISTRIBUTION
# ==========================================================

def plot_categorical(df, column):

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.countplot(
        data=df,
        x=column,
        ax=ax
    )

    ax.set_title(f"Distribution of {column}")
    ax.tick_params(axis="x", rotation=45)

    return fig


# ==========================================================
# 4. BIVARIATE ANALYSIS - SCATTER PLOT
# ==========================================================

def plot_scatter(df, column1, column2):

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.scatterplot(
        data=df,
        x=column1,
        y=column2,
        ax=ax
    )

    ax.set_title(f"{column1} vs {column2}")

    return fig


# ==========================================================
# 5. BIVARIATE ANALYSIS - CORRELATION
# ==========================================================

def get_correlation(df, column1, column2):

    data = df[[column1, column2]].dropna()

    return {
        "Pearson Correlation": data[column1].corr(
            data[column2],
            method="pearson"
        ),
        "Spearman Correlation": data[column1].corr(
            data[column2],
            method="spearman"
        )
    }


# ==========================================================
# 6. BIVARIATE ANALYSIS - CATEGORICAL VS NUMERICAL
# ==========================================================

def plot_group_comparison(df, categorical_column, numerical_column):

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.boxplot(
        data=df,
        x=categorical_column,
        y=numerical_column,
        ax=ax
    )

    ax.set_title(
        f"{numerical_column} across {categorical_column}"
    )

    ax.tick_params(axis="x", rotation=45)

    return fig


# ==========================================================
# 7. BIVARIATE ANALYSIS - CONTINGENCY TABLE
# ==========================================================

def get_contingency_table(df, column1, column2):

    return pd.crosstab(
        df[column1],
        df[column2]
    )


# ==========================================================
# 8. MULTIVARIATE ANALYSIS - CORRELATION MATRIX
# ==========================================================

def get_correlation_matrix(df):

    numeric_df = df.select_dtypes(
        include="number"
    )

    if numeric_df.shape[1] < 2:
        return None

    return numeric_df.corr()


# ==========================================================
# 9. MULTIVARIATE ANALYSIS - CORRELATION HEATMAP
# ==========================================================

def plot_correlation(df):

    correlation = get_correlation_matrix(df)

    if correlation is None:
        return None

    fig, ax = plt.subplots(figsize=(10, 7))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="Greens",
        fmt=".2f",
        ax=ax
    )

    ax.set_title("Correlation Heatmap")

    return fig


# ==========================================================
# 10. MULTIVARIATE ANALYSIS - PAIRPLOT
# ==========================================================

def plot_pairplot(df, columns):

    selected_columns = columns[:5]

    if len(selected_columns) < 2:
        return None

    plot_df = df[selected_columns].dropna()

    if plot_df.empty:
        return None

    pair_plot = sns.pairplot(
        plot_df,
        diag_kind="hist"
    )

    pair_plot.figure.suptitle(
        "Multivariate Pairplot",
        y=1.02
    )

    return pair_plot.figure