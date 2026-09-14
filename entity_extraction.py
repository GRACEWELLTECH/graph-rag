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
You are an information extraction system.

Extract the important entities from the following text.

Return ONLY valid JSON in this format:

[
  {{
    "name": "entity name",
    "type": "entity type"
  }}
]

Allowed entity types:
PERSON
ORGANIZATION
LOCATION
TECHNOLOGY
CONCEPT
FACILITY
ACTIVITY
OTHER

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

response = chain.invoke({"text": text})

print("\n===== STRUCTURED ENTITIES =====\n")
print(response.content)