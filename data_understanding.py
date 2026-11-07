
import pandas as pd


# ==========================================================
# 1. DATASET OVERVIEW
# ==========================================================

def get_dataset_overview(df):

    memory = df.memory_usage(deep=True).sum()

    overview = {
        "Total Rows": df.shape[0],
        "Total Columns": df.shape[1],
        "Duplicate Rows": df.duplicated().sum(),
        "Memory Usage (MB)": round(
            memory / (1024 ** 2), 2
        )
    }

    return overview


# ==========================================================
# 2. COLUMN INFORMATION
# ==========================================================

def get_column_information(df):

    column_info = pd.DataFrame({

        "Column Name": df.columns,

        "Data Type": df.dtypes.astype(str).values,

        "Non-Null Count": df.notnull().sum().values,

        "Missing Values": df.isnull().sum().values,

        "Missing Percentage": (
            df.isnull().mean().values * 100
        ).round(2)

    })

    return column_info


# ==========================================================
# 3. COLUMN TYPE IDENTIFICATION
# ==========================================================

def get_column_types(df):

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    return numeric_columns, categorical_columns


# ==========================================================
# 4. DESCRIPTIVE STATISTICS
# ==========================================================

def get_descriptive_statistics(df):

    return df.describe(
        include="all"
    ).T
