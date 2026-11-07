import os
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==========================================================
# 1. INITIALIZE GEMINI MODEL
# ==========================================================

def get_gemini_model():

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing.")

    model_name = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)

    model = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.2
    )

    return model


# ==========================================================
# 2. CREATE AI PROMPT
# ==========================================================

def create_analysis_prompt():

    prompt = ChatPromptTemplate.from_messages([

        (
            "system",
            """
            You are StatAgent AI, an expert statistical
            and machine learning assistant.

            Your responsibilities:

            1. Explain analytical results.
            2. Interpret statistical test results.
            3. Identify important patterns.
            4. Explain machine learning recommendations.
            5. Provide meaningful analytical insights.

            Important instructions:

            - Use simple and understandable language.
            - Do not invent numerical results.
            - Base your explanations on the provided data.
            - Distinguish statistical evidence from assumptions.
            - Provide practical recommendations.
            """
        ),

        (
            "human",
            """
            Analytical Results:

            {analysis_results}

            User Question:

            {user_question}

            Analyze the results and provide a clear explanation.
            """
        )

    ])

    return prompt


# ==========================================================
# 3. GENERATE AI INSIGHTS
# ==========================================================

def generate_insights(analysis_results, user_question=None):

    model = get_gemini_model()

    prompt = create_analysis_prompt()

    chain = prompt | model | StrOutputParser()

    formatted_results = json.dumps(
        analysis_results,
        indent=2,
        default=str
    )

    response = chain.invoke({

        "analysis_results": formatted_results,

        "user_question": (
            user_question
            if user_question
            else "Explain the analytical results and provide insights."
        )

    })

    return response