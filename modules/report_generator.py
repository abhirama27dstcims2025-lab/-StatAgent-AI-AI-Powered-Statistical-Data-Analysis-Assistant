import pandas as pd
from io import BytesIO


# ==========================================================
# 1. GENERATE EXCEL REPORT
# ==========================================================

def generate_excel_report(
    overview,
    column_info,
    descriptive_stats,
    statistical_results=None,
    ai_insights=None
):

    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:

        # Dataset overview
        overview_df = pd.DataFrame(
            list(overview.items()),
            columns=["Metric", "Value"]
        )

        overview_df.to_excel(
            writer,
            sheet_name="Dataset Overview",
            index=False
        )

        # Column information
        column_info.to_excel(
            writer,
            sheet_name="Column Information",
            index=False
        )

        # Descriptive statistics
        descriptive_stats.to_excel(
            writer,
            sheet_name="Descriptive Statistics"
        )

        # Statistical results
        if statistical_results:

            statistical_df = pd.DataFrame(
                statistical_results
            )

            statistical_df.to_excel(
                writer,
                sheet_name="Statistical Results",
                index=False
            )

        # AI insights
        if ai_insights:

            insights_df = pd.DataFrame({
                "AI Insights": [ai_insights]
            })

            insights_df.to_excel(
                writer,
                sheet_name="AI Insights",
                index=False
            )

    output.seek(0)

    return output