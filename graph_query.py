import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


class GraphQuery:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    def find_entity(self, entity_name):
        query = """
        MATCH (e:Entity)
        WHERE toLower(e.name) = toLower($entity_name)

        OPTIONAL MATCH (e)-[r]-(related:Entity)

        RETURN
            e.name AS entity,
            e.type AS entity_type,
            type(r) AS relationship,
            related.name AS related_entity,
            related.type AS related_type
        """

        with self.driver.session() as session:
            results = session.run(
                query,
                entity_name=entity_name
            )

            return [record.data() for record in results]


if __name__ == "__main__":

    graph = GraphQuery()

    try:
        entity = "Adani Kattupalli Port"

        results = graph.find_entity(entity)

        print("\n" + "=" * 60)
        print(f"GRAPH RESULTS FOR: {entity}")
        print("=" * 60)

        for result in results:
            print(
                f"\n{result['entity']}"
                f" --[{result['relationship']}]--> "
                f"{result['related_entity']}"
            )

    finally:
        graph.close()