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
# 3. GRAPH RAG
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
    # CLOSE CONNECTION
    # =====================================================

    def close(self):
        self.driver.close()

    # =====================================================
    # SEARCH GRAPH
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

            return list(dict.fromkeys(context))

    # =====================================================
    # SEARCH MULTIPLE TERMS
    # =====================================================

    def search_multiple_entities(self, terms):

        all_context = []

        for term in terms:

            results = self.search_graph(term)

            all_context.extend(results)

        return list(dict.fromkeys(all_context))

    # =====================================================
    # GENERATE ANSWER
    # =====================================================

    def generate_answer(self, question, graph_context):

        context_text = "\n".join(graph_context)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a Knowledge Graph RAG assistant.

Answer the question using ONLY the supplied
Knowledge Graph Context.

Rules:

1. Do not invent facts.
2. Do not use outside knowledge.
3. If the context does not contain enough information,
   say that the information is not available.
4. Give a concise answer.
5. Use the relationships in the graph to understand
   the answer.

Knowledge Graph Context:

{context}
"""
                ),
                (
                    "human",
                    "Question: {question}"
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
# 4. MAIN
# =========================================================

if __name__ == "__main__":

    rag = GraphRAG()

    try:

        # -------------------------------------------------
        # Question
        # -------------------------------------------------

        question = "What technology is used in port operations?"

        # -------------------------------------------------
        # Search terms
        # -------------------------------------------------

        search_terms = [
            "Port Operations",
            "technology",
            "machinery",
            "monitoring",
            "communication"
        ]

        # -------------------------------------------------
        # Retrieve graph context
        # -------------------------------------------------

        graph_context = rag.search_multiple_entities(
            search_terms
        )

        print("\n" + "=" * 70)
        print("GRAPH RAG")
        print("=" * 70)

        print("\nQUESTION:")
        print(question)

        print("\n" + "-" * 70)
        print("RETRIEVED GRAPH CONTEXT")
        print("-" * 70)

        if graph_context:

            for item in graph_context:
                print(item)

        else:

            print("No graph context found.")

        # -------------------------------------------------
        # Generate answer
        # -------------------------------------------------

        print("\n" + "-" * 70)
        print("GENERATED ANSWER")
        print("-" * 70)

        if graph_context:

            answer = rag.generate_answer(
                question,
                graph_context
            )

            print(answer)

        else:

            print(
                "No relevant information found "
                "in the knowledge graph."
            )

        print("\n" + "=" * 70)

    finally:

        rag.close()