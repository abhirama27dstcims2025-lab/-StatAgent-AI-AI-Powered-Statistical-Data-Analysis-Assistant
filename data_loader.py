
import pandas as pd


# ==========================================================
# LOAD CSV AND EXCEL DATASETS
# ==========================================================

def load_data(uploaded_file):

    if uploaded_file is None:
        return None

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):

        df = pd.read_csv(uploaded_file)

    elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):

        df = pd.read_excel(uploaded_file)

    else:

        raise ValueError(
            "Unsupported file format. Please upload CSV or Excel."
        )

    # Clean column names
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    return df
