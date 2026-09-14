import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# 1. LOAD ENVIRONMENT
# =========================================================

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# =========================================================
# 2. CHECK CONFIGURATION
# =========================================================

if not NEO4J_URI:
    raise ValueError("NEO4J_URI is missing from .env")

if not NEO4J_USERNAME:
    raise ValueError("NEO4J_USERNAME is missing from .env")

if not NEO4J_PASSWORD:
    raise ValueError("NEO4J_PASSWORD is missing from .env")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing from .env")


# =========================================================
# 3. GRAPH RAG CLASS
# =========================================================

class GraphRAG:

    def __init__(self):

        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )

        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )


    # =====================================================
    # CLOSE
    # =====================================================

    def close(self):
        self.driver.close()


    # =====================================================
    # ANALYZE QUESTION
    # =====================================================

    def analyze_question(self, question):

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a Knowledge Graph query analyzer.

Identify the important entities, concepts,
technologies, organizations, locations, activities,
or other terms from the user's question.

Return ONLY a comma-separated list.

Do not answer the question.

Example:

Question:
What technology is used in port operations?

Output:
technology, Port Operations

Question:
Who manages Adani Kattupalli Port?

Output:
Adani Kattupalli Port, manages
"""
                ),
                (
                    "human",
                    "{question}"
                )
            ]
        )

        chain = prompt | self.llm

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


    # =====================================================
    # SEARCH NEO4J
    # =====================================================

    def search_graph(self, search_term):

        query = """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($search_term)

        OPTIONAL MATCH (e)-[r]-(related:Entity)

        RETURN
            e.name AS entity,
            e.type AS entity_type,
            type(r) AS relationship,
            related.name AS related_entity,
            related.type AS related_type
        """

        with self.driver.session() as session:

            result = session.run(
                query,
                search_term=search_term
            )

            context = []

            for record in result:

                if record["related_entity"]:

                    context.append(
                        f"{record['entity']} "
                        f"--[{record['relationship']}]--> "
                        f"{record['related_entity']}"
                    )

            return context


    # =====================================================
    # RETRIEVE CONTEXT FOR ALL TERMS
    # =====================================================

    def retrieve_context(self, terms):

        all_context = []

        for term in terms:

            results = self.search_graph(term)

            all_context.extend(results)

        return list(dict.fromkeys(all_context))


    # =====================================================
    # GENERATE ANSWER
    # =====================================================

    def generate_answer(self, question, context):

        context_text = "\n".join(context)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a Knowledge Graph RAG assistant.

Answer the user's question using ONLY the
Knowledge Graph Context.

Rules:

1. Do not invent facts.
2. Do not use outside knowledge.
3. Use the relationships in the graph.
4. If the answer is not present in the context,
   say that the information is not available
   in the knowledge graph.
5. Give a concise factual answer.

Knowledge Graph Context:

{context}
"""
                ),
                (
                    "human",
                    "{question}"
                )
            ]
        )

        chain = prompt | self.llm

        response = chain.invoke(
            {
                "context": context_text,
                "question": question
            }
        )

        return response.content


# =========================================================
# 4. INTERACTIVE APPLICATION
# =========================================================

if __name__ == "__main__":

    rag = GraphRAG()

    try:

        print("\n" + "=" * 70)
        print("             KNOWLEDGE GRAPH RAG")
        print("=" * 70)

        print("\nType your question.")
        print("Type 'exit' to quit.")

        while True:

            print("\n" + "-" * 70)

            question = input("Enter your question: ").strip()

            if question.lower() in ["exit", "quit"]:

                print("\nExiting Graph RAG...")
                break

            if not question:

                print("Please enter a question.")
                continue


            # =================================================
            # QUESTION ANALYSIS
            # =================================================

            terms = rag.analyze_question(question)

            print("\nQUESTION ANALYSIS")
            print("-" * 70)

            for term in terms:
                print(f"• {term}")


            # =================================================
            # GRAPH RETRIEVAL
            # =================================================

            context = rag.retrieve_context(terms)

            print("\nRETRIEVED GRAPH CONTEXT")
            print("-" * 70)

            if context:

                for item in context:
                    print(item)

            else:

                print("No relevant graph context found.")


            # =================================================
            # ANSWER
            # =================================================

            print("\nGENERATED ANSWER")
            print("-" * 70)

            if context:

                answer = rag.generate_answer(
                    question,
                    context
                )

                print(answer)

            else:

                print(
                    "The information is not available "
                    "in the knowledge graph."
                )


    finally:

        rag.close()