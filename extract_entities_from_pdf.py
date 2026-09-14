import os
import json
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# --------------------------------------------------
# 1. Load PDF
# --------------------------------------------------

PDF_PATH = "documents/sample.pdf"

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print(f"✅ Pages loaded: {len(documents)}")


# --------------------------------------------------
# 2. Split PDF into chunks
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"✅ Chunks created: {len(chunks)}")


# --------------------------------------------------
# 3. Create LLM
# --------------------------------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# --------------------------------------------------
# 4. Entity extraction prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template("""
You are a knowledge graph entity extraction system.

Extract the important entities from the text.

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

Rules:

1. Extract meaningful entities only.
2. Do not extract complete sentences.
3. Avoid duplicate entities.
4. Preserve the entity name as it appears in the text.
5. Do not invent information.

TEXT:

{text}
""")


chain = prompt | llm


# --------------------------------------------------
# 5. Extract entities from first 5 chunks
# --------------------------------------------------

for i, chunk in enumerate(chunks[:5]):

    print("\n" + "=" * 60)
    print(f"CHUNK {i + 1}")
    print("=" * 60)

    print(f"Page: {chunk.metadata.get('page_label', 'Unknown')}")

    text = chunk.page_content

    response = chain.invoke({
        "text": text
    })

    print("\nEntities:")

    try:
        entities = json.loads(response.content)

        for entity in entities:
            print(
                f"- {entity['name']} "
                f"[{entity['type']}]"
            )

    except json.JSONDecodeError:
        print(response.content)