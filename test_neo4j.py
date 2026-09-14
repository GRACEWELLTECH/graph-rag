import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(
    URI,
    auth=(USERNAME, PASSWORD)
)

try:
    driver.verify_connectivity()
    print("✅ Connected to Neo4j AuraDB!")

    with driver.session() as session:

        result = session.run("""
            MATCH (d:Document)-[:MENTIONS]->(e:Entity)
            RETURN d.name AS document,
                   e.name AS entity,
                   e.type AS entity_type
        """)

        for record in result:
            print(
                f"Document: {record['document']}"
                f" | Entity: {record['entity']}"
                f" | Type: {record['entity_type']}"
            )

finally:
    driver.close()