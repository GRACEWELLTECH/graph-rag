import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

prompt = ChatPromptTemplate.from_template("""
You are a knowledge graph extraction system.

Extract entities and relationships from the text.

Return ONLY valid JSON using this structure:

{{
  "entities": [
    {{
      "name": "entity name",
      "type": "entity type"
    }}
  ],
  "relationships": [
    {{
      "source": "source entity name",
      "relationship": "RELATIONSHIP_TYPE",
      "target": "target entity name"
    }}
  ]
}}

Allowed entity types:

PERSON
ORGANIZATION
LOCATION
TECHNOLOGY
CONCEPT
FACILITY
ACTIVITY
OTHER

Use short, clear relationship types such as:

MANAGED_BY
LOCATED_IN
PART_OF
USES
HANDLES
SUPPORTS
RELATED_TO
HAS_ACTIVITY
HAS_ROLE

Rules:

1. Extract only information supported by the text.
2. Do not invent relationships.
3. Use the exact entity names from the text where possible.
4. Avoid duplicate entities.
5. Return ONLY JSON.

TEXT:

{text}
""")

text = """
Adani Kattupalli Port, managed by the Adani Ports and
Special Economic Zone (APSEZ), is a world-class facility
known for its advanced infrastructure, efficient management
systems, and sustainable operations. It handles diverse
cargo types and plays a critical role in India's maritime
logistics network.

During the visit, students learned about the port's
operational structure, including cargo management,
container handling, coordination among departments,
and safety protocols.
"""

chain = prompt | llm

response = chain.invoke({
    "text": text
})

print("\n===== KNOWLEDGE GRAPH EXTRACTION =====\n")
print(response.content)