
import streamlit as st
import pandas as pd

from dotenv import load_dotenv

from modules.data_loader import load_data

from modules.data_understanding import (
    get_dataset_overview,
    get_column_information,
    get_descriptive_statistics
)

from modules.EDA import (
    plot_distribution,
    plot_boxplot,
    plot_categorical,
    plot_scatter,
    get_correlation,
    plot_group_comparison,
    get_contingency_table,
    get_correlation_matrix,
    plot_correlation,
    plot_pairplot
)

from modules.Statistical_advisor import (
    shapiro_test,
    chi_square_test,
    recommend_statistical_tests
)

from modules.ML_advisor import (
    detect_problem_type,
    recommend_ml_models
)



from modules.ai_agent import generate_insights

from modules.report_generator import generate_excel_report


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

load_dotenv()

st.set_page_config(
    page_title="StatAgent AI",
    page_icon="📊",
    layout="wide"
)


# ==========================================================
# 2. SESSION STATE
# ==========================================================

if "statistical_results" not in st.session_state:
    st.session_state["statistical_results"] = {}

if "ai_insights" not in st.session_state:
    st.session_state["ai_insights"] = None


# ==========================================================
# 3. APPLICATION HEADER
# ==========================================================

st.title("📊 StatAgent AI")

st.subheader("AI-Powered Data Analysis Assistant")

st.write(
    "Upload your dataset to perform data understanding, "
    "data quality analysis, EDA, statistical analysis, "
    "machine learning recommendations and AI-powered insights."
)


# ==========================================================
# 4. DATASET UPLOAD
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload your CSV or Excel dataset",
    type=["csv", "xlsx", "xls"]
)


if uploaded_file is not None:

    try:

        # ==================================================
        # 5. LOAD DATASET
        # ==================================================

        df = load_data(uploaded_file)

        if df is None or df.empty:

            st.error("Uploaded dataset is empty.")

            st.stop()

        st.success("Dataset uploaded successfully!")


        # Reset previous results when a different dataset is uploaded
        current_file = uploaded_file.name

        if st.session_state.get("current_file") != current_file:

            st.session_state["current_file"] = current_file
            st.session_state["statistical_results"] = {}
            st.session_state["ai_insights"] = None


        # ==================================================
        # 6. DATASET PREVIEW
        # ==================================================

        st.header("📁 Dataset Preview")

        st.dataframe(
            df.head(),
            use_container_width=True
        )

        st.write("Dataset Shape:", df.shape)


        # ==================================================
        # 7. DATA UNDERSTANDING
        # ==========================================================

        st.header("📋 Data Understanding")

        overview = get_dataset_overview(df)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Rows",
            overview["Total Rows"]
        )

        col2.metric(
            "Total Columns",
            overview["Total Columns"]
        )

        col3.metric(
            "Duplicate Rows",
            overview["Duplicate Rows"]
        )

        col4.metric(
            "Memory Usage",
            f'{overview["Memory Usage (MB)"]} MB'
        )


        # Column Information

        st.subheader("Column Information")

        column_info = get_column_information(df)

        st.dataframe(
            column_info,
            use_container_width=True
        )


        # Descriptive Statistics

        st.subheader("Descriptive Statistics")

        descriptive_stats = get_descriptive_statistics(df)

        st.dataframe(
            descriptive_stats,
            use_container_width=True
        )


        # ==================================================
        # 8. DATA QUALITY ASSURANCE
        # ==================================================

        st.header("🔍 Data Quality Assurance")


        # Missing Values

        st.subheader("Missing Value Analysis")

        missing_df = pd.DataFrame({

            "Column": df.columns,

            "Missing Values": df.isnull().sum().values,

            "Missing Percentage": (
                df.isnull().mean().values * 100
            ).round(2)

        })

        st.dataframe(
            missing_df,
            use_container_width=True
        )


        # Duplicate Rows

        st.subheader("Duplicate Row Analysis")

        duplicate_count = df.duplicated().sum()

        st.metric(
            "Duplicate Rows",
            duplicate_count
        )


        # Constant Columns

        st.subheader("Constant Column Detection")

        constant_columns = []

        for column in df.columns:

            if df[column].nunique(dropna=False) <= 1:

                constant_columns.append(column)


        if constant_columns:

            st.warning("Constant columns detected.")

            st.write(constant_columns)

        else:

            st.success("No constant columns detected.")


        # High Cardinality

        st.subheader("High Cardinality Detection")

        high_cardinality_results = []

        for column in df.select_dtypes(
            include=["object", "category"]
        ).columns:

            unique_count = df[column].nunique()

            unique_percentage = (
                unique_count / len(df) * 100
            )

            if unique_percentage > 50 and unique_count > 10:

                high_cardinality_results.append({

                    "Column": column,

                    "Unique Values": unique_count,

                    "Unique Percentage": round(
                        unique_percentage, 2
                    )

                })


        high_cardinality_df = pd.DataFrame(
            high_cardinality_results,
            columns=[
                "Column",
                "Unique Values",
                "Unique Percentage"
            ]
        )


        if not high_cardinality_df.empty:

            st.warning("High-cardinality columns detected.")

            st.dataframe(
                high_cardinality_df,
                use_container_width=True
            )

        else:

            st.success("No high-cardinality columns detected.")


        # ==================================================
        # 9. IDENTIFY COLUMN TYPES
        # ==================================================

        numeric_columns = df.select_dtypes(
            include="number"
        ).columns.tolist()

        categorical_columns = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()


        # ==================================================
        # 10. EXPLORATORY DATA ANALYSIS
        # ==================================================

        st.header("📊 Exploratory Data Analysis")


        # --------------------------------------------------
        # 10.1 UNIVARIATE ANALYSIS
        # --------------------------------------------------

        st.subheader("10.1 Univariate Analysis")


        # Numerical Distribution

        st.markdown("**Numerical Variable Distribution**")

        if numeric_columns:

            selected_numeric = st.selectbox(
                "Select numerical column",
                numeric_columns,
                key="univariate_numeric"
            )

            distribution_fig = plot_distribution(
                df,
                selected_numeric
            )

            st.pyplot(distribution_fig)

            st.markdown("**Boxplot / Outlier Visualization**")

            boxplot_fig = plot_boxplot(
                df,
                selected_numeric
            )

            st.pyplot(boxplot_fig)

        else:

            st.info("No numerical columns available.")


        # Categorical Distribution

        st.markdown("**Categorical Variable Distribution**")

        if categorical_columns:

            selected_categorical = st.selectbox(
                "Select categorical column",
                categorical_columns,
                key="univariate_categorical"
            )

            categorical_fig = plot_categorical(
                df,
                selected_categorical
            )

            st.pyplot(categorical_fig)

        else:

            st.info("No categorical columns available.")


        # --------------------------------------------------
        # 10.2 BIVARIATE ANALYSIS
        # --------------------------------------------------

        st.header("🔗 Bivariate Analysis")


        # Numerical vs Numerical

        st.subheader("Numerical vs Numerical")

        if len(numeric_columns) >= 2:

            col1, col2 = st.columns(2)

            with col1:

                numerical_1 = st.selectbox(
                    "Select first numerical variable",
                    numeric_columns,
                    key="bivariate_num1"
                )

            with col2:

                remaining_numeric = [
                    col for col in numeric_columns
                    if col != numerical_1
                ]

                numerical_2 = st.selectbox(
                    "Select second numerical variable",
                    remaining_numeric,
                    key="bivariate_num2"
                )


            scatter_fig = plot_scatter(
                df,
                numerical_1,
                numerical_2
            )

            st.pyplot(scatter_fig)


            correlation_results = get_correlation(
                df,
                numerical_1,
                numerical_2
            )

            col1, col2 = st.columns(2)

            pearson_value = correlation_results[
                "Pearson Correlation"
            ]

            spearman_value = correlation_results[
                "Spearman Correlation"
            ]

            col1.metric(
                "Pearson Correlation",
                f"{pearson_value:.3f}"
                if pd.notna(pearson_value) else "N/A"
            )

            col2.metric(
                "Spearman Correlation",
                f"{spearman_value:.3f}"
                if pd.notna(spearman_value) else "N/A"
            )

        else:

            st.info(
                "At least two numerical columns are required."
            )


        # Categorical vs Numerical

        st.subheader("Categorical vs Numerical")

        if categorical_columns and numeric_columns:

            col1, col2 = st.columns(2)

            with col1:

                selected_cat = st.selectbox(
                    "Select categorical variable",
                    categorical_columns,
                    key="bivariate_cat"
                )

            with col2:

                selected_num = st.selectbox(
                    "Select numerical variable",
                    numeric_columns,
                    key="bivariate_num"
                )


            group_fig = plot_group_comparison(
                df,
                selected_cat,
                selected_num
            )

            st.pyplot(group_fig)


            st.markdown("**Group-wise Summary Statistics**")

            group_summary = df.groupby(
                selected_cat,
                observed=True
            )[selected_num].agg(
                ["count", "mean", "median", "std", "min", "max"]
            )

            st.dataframe(
                group_summary,
                use_container_width=True
            )

        else:

            st.info(
                "Categorical and numerical columns are required."
            )


        # Categorical vs Categorical

        st.subheader("Categorical vs Categorical")

        if len(categorical_columns) >= 2:

            col1, col2 = st.columns(2)

            with col1:

                cat_1 = st.selectbox(
                    "Select first categorical variable",
                    categorical_columns,
                    key="cat_cat_1"
                )

            with col2:

                remaining_categorical = [
                    col for col in categorical_columns
                    if col != cat_1
                ]

                cat_2 = st.selectbox(
                    "Select second categorical variable",
                    remaining_categorical,
                    key="cat_cat_2"
                )


            contingency_table = get_contingency_table(
                df,
                cat_1,
                cat_2
            )

            st.markdown("**Contingency Table**")

            st.dataframe(
                contingency_table,
                use_container_width=True
            )

        else:

            st.info(
                "At least two categorical columns are required."
            )


        # --------------------------------------------------
        # 10.3 MULTIVARIATE ANALYSIS
        # --------------------------------------------------

        st.header("📈 Multivariate Analysis")


        # Correlation Matrix

        st.subheader("Correlation Matrix")

        correlation_matrix = get_correlation_matrix(df)

        if correlation_matrix is not None:

            st.dataframe(
                correlation_matrix,
                use_container_width=True
            )

            correlation_fig = plot_correlation(df)

            if correlation_fig is not None:

                st.pyplot(correlation_fig)

        else:

            st.info(
                "At least two numerical columns are required."
            )


        # Pairplot

        st.subheader("Multivariate Pairplot")

        if len(numeric_columns) >= 2:

            selected_pair_columns = st.multiselect(
                "Select numerical variables for pairplot (maximum 5)",
                numeric_columns,
                default=numeric_columns[
                    :min(3, len(numeric_columns))
                ],
                max_selections=5
            )

            if len(selected_pair_columns) >= 2:

                pairplot_fig = plot_pairplot(
                    df,
                    selected_pair_columns
                )

                if pairplot_fig is not None:

                    st.pyplot(pairplot_fig)

            else:

                st.info(
                    "Select at least two numerical variables."
                )

        else:

            st.info(
                "At least two numerical columns are required."
            )


        # ==================================================
        # 11. STATISTICAL ANALYSIS
        # ==================================================

        st.header("📐 Statistical Analysis")

        st.write(
            "Run statistical tests to investigate distributions "
            "and relationships between variables."
        )


        # --------------------------------------------------
        # 11.1 SHAPIRO-WILK TEST
        # --------------------------------------------------

        st.subheader("Shapiro-Wilk Normality Test")

        if numeric_columns:

            shapiro_column = st.selectbox(
                "Select numerical variable",
                numeric_columns,
                key="shapiro_column"
            )

            if st.button("Run Shapiro-Wilk Test"):

                result = shapiro_test(
                    df,
                    shapiro_column
                )

                if isinstance(result, dict):

                    result["Variable"] = shapiro_column

                    st.session_state[
                        "statistical_results"
                    ]["Shapiro-Wilk"] = result

                else:

                    st.warning(result)


            shapiro_result = st.session_state[
                "statistical_results"
            ].get("Shapiro-Wilk")

            if shapiro_result:

                st.write(
                    "Test Statistic:",
                    round(shapiro_result["Statistic"], 4)
                )

                st.write(
                    "P-value:",
                    round(shapiro_result["P-value"], 4)
                )

                st.info(
                    shapiro_result["Conclusion"]
                )

        else:

            st.info("No numerical columns available.")


        # --------------------------------------------------
        # 11.2 CHI-SQUARE TEST
        # --------------------------------------------------

        st.subheader("Chi-Square Test of Independence")

        if len(categorical_columns) >= 2:

            col1, col2 = st.columns(2)

            with col1:

                chi_column1 = st.selectbox(
                    "Select first categorical variable",
                    categorical_columns,
                    key="chi_column1"
                )

            with col2:

                chi_options = [
                    col for col in categorical_columns
                    if col != chi_column1
                ]

                chi_column2 = st.selectbox(
                    "Select second categorical variable",
                    chi_options,
                    key="chi_column2"
                )


            if st.button("Run Chi-Square Test"):

                result = chi_square_test(
                    df,
                    chi_column1,
                    chi_column2
                )

                if isinstance(result, dict):

                    result["Variables"] = (
                        f"{chi_column1} vs {chi_column2}"
                    )

                    st.session_state[
                        "statistical_results"
                    ]["Chi-Square"] = result

                else:

                    st.warning(result)


            chi_result = st.session_state[
                "statistical_results"
            ].get("Chi-Square")

            if chi_result:

                st.write(
                    "Test Statistic:",
                    round(chi_result["Statistic"], 4)
                )

                st.write(
                    "P-value:",
                    round(chi_result["P-value"], 4)
                )

                st.write(
                    "Degrees of Freedom:",
                    chi_result["Degrees of Freedom"]
                )

                st.info(
                    chi_result["Conclusion"]
                )

        else:

            st.info(
                "At least two categorical columns are required."
            )


        # --------------------------------------------------
        # 11.3 STATISTICAL TEST RECOMMENDATIONS
        # --------------------------------------------------

        st.subheader("Statistical Test Recommendations")

        statistical_target = st.selectbox(
            "Select target variable",
            ["None"] + df.columns.tolist(),
            key="statistical_target"
        )


        if statistical_target == "None":

            statistical_problem = None

        else:

            statistical_problem = detect_problem_type(
                df,
                statistical_target
            )


        predictor_numeric = [
            col for col in numeric_columns
            if col != statistical_target
        ]

        predictor_categorical = [
            col for col in categorical_columns
            if col != statistical_target
        ]


        recommendations = recommend_statistical_tests(
            df,
            None if statistical_target == "None"
            else statistical_target,
            statistical_problem,
            predictor_numeric,
            predictor_categorical
        )


        for test_name, explanation in recommendations:

            st.markdown(f"**{test_name}**")

            st.write(explanation)


        # ==================================================
        # 12. MACHINE LEARNING ANALYSIS
        # ==================================================

        st.header("🤖 Machine Learning Analysis")


        # ML Target Selection

        ml_target = st.selectbox(
            "Select target variable for ML",
            ["None"] + df.columns.tolist(),
            key="ml_target"
        )


        if ml_target == "None":

            problem_type = detect_problem_type(
                df,
                None
            )

        else:

            problem_type = detect_problem_type(
                df,
                ml_target
            )


        st.subheader("Detected ML Problem Type")

        st.success(
            problem_type
        )


        # Model Recommendations

        st.subheader("Recommended Machine Learning Models")

        ml_recommendations = recommend_ml_models(
            problem_type
        )


        if ml_recommendations:

            for model_name, explanation in ml_recommendations:

                st.markdown(f"**{model_name}**")

                st.write(explanation)

        else:

            st.info(
                "No model recommendations available."
            )


        # ==================================================
        # 13. GEMINI AI INSIGHTS
        # ==================================================

        st.header("✨ AI-Powered Insights")

        st.write(
            "Gemini interprets the results calculated by Python "
            "and provides understandable analytical insights."
        )


        user_question = st.text_area(
            "Ask a question about your dataset",
            placeholder=(
                "Example: Explain the important patterns "
                "and suggest possible next steps."
            ),
            key="ai_question"
        )


        if st.button("Generate AI Insights"):

            analysis_results = {

                "Dataset Overview": overview,

                "Column Information": column_info.to_dict(
                    orient="records"
                ),

                "Descriptive Statistics": descriptive_stats.to_dict(),

                "Data Quality": {

                    "Missing Values": missing_df.to_dict(
                        orient="records"
                    ),

                    "Duplicate Rows": duplicate_count,

                    "Constant Columns": constant_columns,

                    "High Cardinality": high_cardinality_df.to_dict(
                        orient="records"
                    )

                },

                "Correlation Matrix": (
                    correlation_matrix.to_dict()
                    if correlation_matrix is not None
                    else "Not available"
                ),

                "Statistical Test Results": list(
                    st.session_state[
                        "statistical_results"
                    ].values()
                ),

                "Statistical Recommendations": recommendations,

                "ML Problem Type": problem_type,

                "ML Model Recommendations": ml_recommendations

            }


            try:

                with st.spinner(
                    "Gemini is analyzing your dataset results..."
                ):

                    ai_response = generate_insights(
                        analysis_results,
                        user_question
                    )

                st.session_state["ai_insights"] = ai_response

                st.success("AI insights generated successfully!")

            except Exception as e:

                st.error(
                    f"Unable to generate AI insights: {e}"
                )


        if st.session_state["ai_insights"]:

            st.subheader("AI Analysis Report")

            st.markdown(
                st.session_state["ai_insights"]
            )


        # ==================================================
        # 14. EXCEL REPORT GENERATION
        # ==================================================

        st.header("📄 Download Analytical Report")


        statistical_results = list(
            st.session_state[
                "statistical_results"
            ].values()
        )


        excel_file = generate_excel_report(

            overview=overview,

            column_info=column_info,

            descriptive_stats=descriptive_stats,

            statistical_results=statistical_results,

            ai_insights=st.session_state["ai_insights"]

        )


        st.download_button(

            label="📥 Download Excel Report",

            data=excel_file,

            file_name="StatAgent_AI_Analytical_Report.xlsx",

            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            )

        )


    except Exception as e:

        st.error(
            f"An error occurred while processing the dataset: {e}"
        )

