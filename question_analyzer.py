import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# LOAD ENVIRONMENT
# =========================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing from .env")


# =========================================================
# LLM
# =========================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# =========================================================
# PROMPT
# =========================================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a Knowledge Graph query analyzer.

Your task is to identify the important entities,
concepts, technologies, organizations, locations,
activities, or other terms from the user's question.

Return ONLY a comma-separated list of search terms.

Do not answer the question.

Example:

Question:
What technology is used in port operations?

Output:
technology, Port Operations

Example:

Question:
Who manages Adani Kattupalli Port?

Output:
Adani Kattupalli Port, manages

Do not add explanations.
"""
        ),
        (
            "human",
            "{question}"
        )
    ]
)


# =========================================================
# ANALYZE QUESTION
# =========================================================

def analyze_question(question):

    chain = prompt | llm

    response = chain.invoke(
        {
            "question": question
        }
    )

    text = response.content.strip()

    terms = [
        term.strip()
        for term in text.split(",")
        if term.strip()
    ]

    return terms


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    question = "What technology is used in port operations?"

    print("\n" + "=" * 70)
    print("QUESTION ANALYZER")
    print("=" * 70)

    print("\nQUESTION:")
    print(question)

    terms = analyze_question(question)

    print("\nSEARCH TERMS:")

    for term in terms:
        print(f"• {term}")