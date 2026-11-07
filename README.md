# 📊 StatAgent AI – AI-Powered Statistical Data Analysis Assistant

## 🚀 Project Overview

StatAgent AI is an AI-powered data analysis assistant developed using Python and Streamlit. It helps users understand datasets, perform exploratory data analysis (EDA), conduct statistical analysis, and explore machine learning approaches through an interactive interface.

The application combines statistical methods, data analysis techniques, machine learning recommendations, and Generative AI to simplify the data analysis workflow.

## 🎯 Project Objectives

- Automate essential data exploration tasks.
- Understand dataset structure and data quality.
- Perform descriptive statistical analysis.
- Identify patterns and relationships through EDA.
- Support statistical testing and interpretation.
- Recommend suitable machine learning approaches.
- Generate AI-powered analytical insights and reports.

## ✨ Key Features

### 1. Dataset Upload and Understanding
- Upload datasets through the Streamlit interface.
- Explore dataset dimensions, columns, and data types.
- Display sample records and dataset information.

### 2. Data Quality Analysis
- Detect missing values.
- Identify duplicate records.
- Detect constant columns.
- Identify high-cardinality columns.
- Examine potential data quality issues.

### 3. Exploratory Data Analysis (EDA)
- Descriptive statistics.
- Univariate analysis.
- Bivariate analysis.
- Histograms and box plots.
- Explore distributions and relationships between variables.

### 4. Statistical Analysis
- Statistical test recommendations.
- Normality testing.
- Statistical interpretation and insights.

### 5. Machine Learning Advisor
- Assist in identifying suitable machine learning approaches.
- Provide model recommendations based on dataset characteristics.

### 6. AI-Powered Insights
- Generate analytical insights using Google's Gemini API.
- Provide natural-language explanations of analytical results.

### 7. Report Generation
- Summarize important analytical findings.
- Generate structured data analysis reports.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| SciPy | Statistical analysis |
| Scikit-learn | Machine learning |
| Google Gemini API | AI-powered insights |

## 📁 Project Structure

```text
StatAgentAI/
│
├── app.py
├── config.py
├── requirements.txt
│
├── modules/
│   ├── data_loader.py
│   ├── data_understanding.py
│   ├── EDA.py
│   ├── statistical_advisor.py
│   ├── ML_advisor.py
│   ├── ai_agent.py
│   └── report_generator.py
│
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/StatAgent-AI.git
```

### 2. Navigate to the project directory

```bash
cd StatAgent-AI
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

Windows:

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure Gemini API

Create a `.env` file in the project directory and add:

```env
GEMINI_API_KEY=your_api_key_here
```

Never upload your actual API key to GitHub.

### 7. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🔮 Future Improvements

- Advanced statistical hypothesis testing.
- Automated machine learning model training and evaluation.
- Feature importance and explainable AI using SHAP.
- Advanced model comparison and optimization.
- Downloadable PDF analysis reports.
- More advanced AI-driven analytical recommendations.

## 👨‍💻 Author

**Abhijit Waghmode**

M.Sc. Statistics and Computing  
DST-CIMS, Banaras Hindu University (BHU)

**Areas of Interest:** Data Analytics, Statistical Modeling, Machine Learning, and Artificial Intelligence.

---

⭐ If you find this project interesting, feel free to explore the repository and share your feedback!

## 🔗 Project Links

- **Live Application:** [Try StatAgent AI](YOUR_STREAMLIT_LINK)



