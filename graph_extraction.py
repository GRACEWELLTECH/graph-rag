import os
import json
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(" OPENAI_API_KEY not found in .env file")


# ============================================================
# 2. PDF CONFIGURATION
# ============================================================

PDF_PATH = "documents/sample.pdf"

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(
        f" PDF not found: {PDF_PATH}"
    )


# ============================================================
# 3. LOAD PDF
# ============================================================

print("\n" + "=" * 70)
print("STEP 1 - LOADING PDF")
print("=" * 70)

loader = PyPDFLoader(PDF_PATH)

documents = loader.load()

print(f"✅ Pages loaded: {len(documents)}")


# ============================================================
# 4. SPLIT PDF INTO CHUNKS
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 - CREATING CHUNKS")
print("=" * 70)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"✅ Chunks created: {len(chunks)}")


# ============================================================
# 5. CREATE LLM
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 - INITIALIZING LLM")
print("=" * 70)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

print("✅ LLM initialized")


# ============================================================
# 6. KNOWLEDGE GRAPH EXTRACTION PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_template("""
You are a high-precision Knowledge Graph extraction system.

Analyze the supplied text and extract:

1. Important entities
2. Meaningful relationships between those entities

Return ONLY valid JSON in this exact structure:

{{
  "entities": [
    {{
      "name": "entity name",
      "type": "entity type"
    }}
  ],
  "relationships": [
    {{
      "source": "source entity",
      "relationship": "RELATIONSHIP_TYPE",
      "target": "target entity"
    }}
  ]
}}


============================================================
ALLOWED ENTITY TYPES
============================================================

PERSON
GROUP
ORGANIZATION
LOCATION
FACILITY
TECHNOLOGY
ACTIVITY
CONCEPT
EVENT
DOCUMENT
OTHER


============================================================
ALLOWED RELATIONSHIP TYPES
============================================================

PART_OF
LOCATED_IN
MANAGED_BY
MENTIONS
USES
HANDLES
SUPPORTS
LEARNED_ABOUT
PARTICIPATED_IN
WORKS_AT
VISITS
HAS_ACTIVITY
HAS_ROLE


============================================================
EXTRACTION RULES
============================================================

1. Extract meaningful entities only.

2. Do not extract complete sentences as entities.

3. Do not invent entities.

4. Do not invent relationships.

5. Avoid duplicate entities within the same chunk.

6. Preserve the entity name as it appears in the source text
   wherever possible.

7. Every relationship must be explicitly supported by the text.

8. Generic words such as "students", "technology",
   "engineering", and "logistics" should not automatically
   become entities unless they represent an important concept
   in the context.

9. "students" should normally be classified as GROUP rather
   than PERSON.

10. Use LEARNED_ABOUT when students learn about a subject,
    system, technology, facility, or activity.

11. Use VISITS when students visit a facility or location.

12. Use HANDLES when an organization or facility handles
    cargo, services, operations, etc.

13. Use HAS_ACTIVITY only for genuine activities.

14. Use PART_OF when one organization, department, unit,
    or component belongs to another organization or system.

15. Use LOCATED_IN for geographic relationships.

16. Use MANAGED_BY when an organization or facility is managed
    by another organization.

17. Use WORKS_AT when a person is explicitly associated with
    an organization or workplace.

18. Use HAS_ROLE when a person's role or designation is
    explicitly stated.

19. Do not create a relationship merely because two entities
    appear in the same sentence.

20. Prefer fewer accurate entities and relationships over many
    uncertain ones.

21. Return ONLY valid JSON.
    
TEXT:

{text}
""")


# ============================================================
# 7. CREATE EXTRACTION CHAIN
# ============================================================

chain = prompt | llm


# ============================================================
# 8. PROCESS CHUNKS
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 - KNOWLEDGE GRAPH EXTRACTION")
print("=" * 70)

all_results = []


# ------------------------------------------------------------
# IMPORTANT:
# For testing, process only the first 5 chunks.
#
# Later we will change:
#
#     chunks[:5]
#
# to:
#
#     chunks
# ------------------------------------------------------------

for i, chunk in enumerate(chunks[:5]):

    print("\n" + "=" * 70)
    print(f"CHUNK {i + 1}")
    print("=" * 70)

    # --------------------------------------------------------
    # Page information
    # --------------------------------------------------------

    page = chunk.metadata.get(
        "page_label",
        chunk.metadata.get(
            "page",
            "Unknown"
        )
    )

    print(f"Page: {page}")

    # --------------------------------------------------------
    # Source text
    # --------------------------------------------------------

    text = chunk.page_content.strip()

    if not text:
        print("Empty chunk - skipped")
        continue

    # --------------------------------------------------------
    # Send chunk to LLM
    # --------------------------------------------------------

    try:

        response = chain.invoke({
            "text": text
        })

        # ----------------------------------------------------
        # Parse JSON
        # ----------------------------------------------------

        data = json.loads(response.content)

        entities = data.get(
            "entities",
            []
        )

        relationships = data.get(
            "relationships",
            []
        )

        # ----------------------------------------------------
        # Display entities
        # ----------------------------------------------------

        print("\nENTITIES:")

        if entities:

            for entity in entities:

                print(
                    f"  • {entity.get('name')} "
                    f"[{entity.get('type')}]"
                )

        else:

            print("  No entities found.")

        # ----------------------------------------------------
        # Display relationships
        # ----------------------------------------------------

        print("\nRELATIONSHIPS:")

        if relationships:

            for relationship in relationships:

                print(
                    f"  • "
                    f"{relationship.get('source')} "
                    f"--[{relationship.get('relationship')}]--> "
                    f"{relationship.get('target')}"
                )

        else:

            print("  No relationships found.")

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        all_results.append({

            "chunk": i + 1,

            "page": page,

            "source": chunk.metadata.get(
                "source",
                PDF_PATH
            ),

            "text": text,

            "entities": entities,

            "relationships": relationships

        })

    except json.JSONDecodeError:

        print("\n⚠️ LLM returned invalid JSON:")

        print(response.content)

    except Exception as e:

        print(
            f"\n❌ Error processing chunk "
            f"{i + 1}: {e}"
        )


# ============================================================
# 9. SAVE EXTRACTION RESULTS
# ============================================================

output_file = "graph_extraction_results.json"

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_results,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# 10. SUMMARY
# ============================================================

total_entities = sum(
    len(result["entities"])
    for result in all_results
)

total_relationships = sum(
    len(result["relationships"])
    for result in all_results
)


print("\n" + "=" * 70)
print("EXTRACTION SUMMARY")
print("=" * 70)

print(
    f"Processed chunks       : {len(all_results)}"
)

print(
    f"Entities extracted     : {total_entities}"
)

print(
    f"Relationships extracted: {total_relationships}"
)

print(
    f"Output file            : {output_file}"
)

print("\n" + "=" * 70)
print("✅ GRAPH EXTRACTION COMPLETED")
print("=" * 70)