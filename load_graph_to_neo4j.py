import os
import json
from dotenv import load_dotenv
from neo4j import GraphDatabase


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")


if not URI:
    raise ValueError("❌ NEO4J_URI not found in .env")

if not USERNAME:
    raise ValueError("❌ NEO4J_USERNAME not found in .env")

if not PASSWORD:
    raise ValueError("❌ NEO4J_PASSWORD not found in .env")


# ============================================================
# 2. LOAD EXTRACTION RESULTS
# ============================================================

INPUT_FILE = "graph_extraction_results.json"

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"❌ File not found: {INPUT_FILE}"
    )


with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    results = json.load(f)


print("\n" + "=" * 70)
print("GRAPH LOADING")
print("=" * 70)

print(
    f"Extraction records loaded: {len(results)}"
)


# ============================================================
# 3. CONNECT TO NEO4J
# ============================================================

driver = GraphDatabase.driver(
    URI,
    auth=(USERNAME, PASSWORD)
)


# ============================================================
# 4. VERIFY CONNECTION
# ============================================================

try:

    driver.verify_connectivity()

    print("✅ Connected to Neo4j AuraDB")


    # ========================================================
    # 5. CREATE DOCUMENT + CHUNK NODES
    # ========================================================

    with driver.session() as session:

        for result in results:

            chunk_number = result["chunk"]
            page = result["page"]
            source = result["source"]
            text = result["text"]

            # ------------------------------------------------
            # Create Document
            # ------------------------------------------------

            session.run(
                """
                MERGE (d:Document {source: $source})
                """,
                source=source
            )

            # ------------------------------------------------
            # Create Chunk
            # ------------------------------------------------

            session.run(
                """
                MERGE (c:Chunk {
                    source: $source,
                    chunk_id: $chunk_id
                })

                SET c.page = $page,
                    c.text = $text

                WITH c

                MATCH (d:Document {source: $source})

                MERGE (d)-[:HAS_CHUNK]->(c)
                """,
                source=source,
                chunk_id=chunk_number,
                page=str(page),
                text=text
            )

            # =================================================
            # 6. CREATE ENTITY NODES
            # =================================================

            for entity in result.get(
                "entities",
                []
            ):

                entity_name = entity.get(
                    "name"
                )

                entity_type = entity.get(
                    "type",
                    "OTHER"
                )

                if not entity_name:
                    continue

                # ---------------------------------------------
                # Create entity
                # ---------------------------------------------

                session.run(
                    """
                    MERGE (e:Entity {
                        name: $name
                    })

                    SET e.type = $type
                    """,
                    name=entity_name,
                    type=entity_type
                )

                # ---------------------------------------------
                # Connect Chunk → Entity
                # ---------------------------------------------

                session.run(
                    """
                    MATCH (c:Chunk {
                        source: $source,
                        chunk_id: $chunk_id
                    })

                    MATCH (e:Entity {
                        name: $name
                    })

                    MERGE (c)-[:MENTIONS]->(e)
                    """,
                    source=source,
                    chunk_id=chunk_number,
                    name=entity_name
                )


            # =================================================
            # 7. CREATE RELATIONSHIPS
            # =================================================

            for relationship in result.get(
                "relationships",
                []
            ):

                source_entity = relationship.get(
                    "source"
                )

                relation_type = relationship.get(
                    "relationship"
                )

                target_entity = relationship.get(
                    "target"
                )

                if not source_entity:
                    continue

                if not relation_type:
                    continue

                if not target_entity:
                    continue


                # ------------------------------------------------
                # Cypher cannot parameterize relationship types.
                #
                # Therefore we validate the relationship type
                # before inserting it.
                # ------------------------------------------------

                allowed_relationships = {
                    "PART_OF",
                    "LOCATED_IN",
                    "MANAGED_BY",
                    "MENTIONS",
                    "USES",
                    "HANDLES",
                    "SUPPORTS",
                    "LEARNED_ABOUT",
                    "PARTICIPATED_IN",
                    "WORKS_AT",
                    "VISITS",
                    "HAS_ACTIVITY",
                    "HAS_ROLE"
                }


                if relation_type not in allowed_relationships:

                    print(
                        f"⚠️ Skipping unknown relationship: "
                        f"{relation_type}"
                    )

                    continue


                query = f"""
                MATCH (source:Entity {{
                    name: $source
                }})

                MATCH (target:Entity {{
                    name: $target
                }})

                MERGE (source)-[:{relation_type}]->(target)
                """


                session.run(
                    query,
                    source=source_entity,
                    target=target_entity
                )


    # ========================================================
    # 8. FINISHED
    # ========================================================

    print("\n" + "=" * 70)
    print("✅ GRAPH LOADING COMPLETED")
    print("=" * 70)


finally:

    driver.close()