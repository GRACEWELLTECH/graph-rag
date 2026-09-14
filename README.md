# HG-RAG — Knowledge Graph Retrieval-Augmented Generation

A prototype **Knowledge Graph RAG (HG-RAG)** system that converts PDF content into a Neo4j knowledge graph and uses graph retrieval plus an LLM to generate grounded answers.

## Architecture

```text
PDF Document
     |
     v
Document Loader
     |
     v
Text Chunking
     |
     v
Entity + Relationship Extraction
     |
     v
graph_extraction_results.json
     |
     v
Neo4j AuraDB
     |
     v
Question Analyzer
     |
     v
Graph Retrieval
     |
     v
Graph Context
     |
     v
GPT-4o-mini
     |
     v
Grounded Answer
```

## Current Status

| Component | Status |
|---|---|
| Neo4j AuraDB | ✅ Working |
| Python → Neo4j connection | ✅ Working |
| PDF loading | ✅ Working |
| Document chunking | ✅ Working |
| Entity extraction | ✅ Working |
| Relationship extraction | ✅ Working |
| JSON extraction output | ✅ Working |
| Neo4j graph loading | ✅ Working |
| Graph visualization | ✅ Working |
| Entity deduplication | ✅ Completed |
| Basic graph retrieval | ✅ Working |
| LLM answer generation | ✅ Working |
| Question analyzer | ✅ Working |
| Interactive Graph RAG | 🟡 Prototype |

## Project Structure

```text
HG-RAG/
│
├── documents/
│   └── sample.pdf
│
├── venv/
│
├── .env
├── .gitignore
│
├── test_neo4j.py
├── document_loader.py
├── chunk_document.py
├── entity_extraction.py
├── relationship_extraction.py
├── graph_extraction.py
├── graph_extraction_results.json
├── load_graph_to_neo4j.py
├── graph_query.py
├── question_analyzer.py
├── graph_rag.py
└── interactive_graph_rag.py
```

## Technologies

- **Python**
- **Neo4j AuraDB**
- **OpenAI GPT-4o-mini**
- **LangChain**
- **PyPDF**
- **python-dotenv**

## Setup

### 1. Create and activate the virtual environment

```powershell
cd C:\Users\Lenovo\GenAI\HG-RAG
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install neo4j python-dotenv
pip install pypdf langchain-community
pip install langchain-text-splitters
pip install langchain-openai
```

### 3. Configure `.env`

Create `.env` in the project root:

```env
NEO4J_URI=neo4j+s://YOUR_INSTANCE.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=YOUR_NEO4J_PASSWORD
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
```

**Never commit `.env` or expose passwords/API keys.**

Recommended `.gitignore`:

```text
venv/
.env
__pycache__/
```

## Pipeline

### 1. Load the PDF

```powershell
python document_loader.py
```

### 2. Chunk the document

```powershell
python chunk_document.py
```

The current prototype uses:

- Chunk size: `1000`
- Chunk overlap: `200`

### 3. Extract entities and relationships

```powershell
python graph_extraction.py
```

The extraction results are saved to:

```text
graph_extraction_results.json
```

### 4. Load the graph into Neo4j

```powershell
python load_graph_to_neo4j.py
```

The graph contains structures such as:

```text
(:Document)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(:Entity)

(:Entity)-[:MANAGED_BY]->(:Entity)
(:Entity)-[:LOCATED_IN]->(:Entity)
(:Entity)-[:USES]->(:Entity)
(:Entity)-[:SUPPORTS]->(:Entity)
```

### 5. Test graph retrieval

```powershell
python graph_query.py
```

### 6. Test question analysis

```powershell
python question_analyzer.py
```

Example:

```text
Question:
What technology is used in port operations?

Search terms:
technology
port operations
```

### 7. Run the interactive Graph RAG

```powershell
python interactive_graph_rag.py
```

Example:

```text
======================================================================
             KNOWLEDGE GRAPH RAG
======================================================================

Type your question.
Type 'exit' to quit.

----------------------------------------------------------------------
Enter your question:
```

Example question:

```text
Who manages Adani Kattupalli Port?
```

Expected type of answer:

```text
Adani Kattupalli Port is managed by Adani Ports and Special Economic Zone.
```

## Knowledge Graph Example

The prototype contains information extracted from a document concerning Adani Kattupalli Port.

Example graph facts include:

```text
Adani Kattupalli Port
        |
        +-- MANAGED_BY --> Adani Ports and Special Economic Zone
        |
        +-- LOCATED_IN --> India
```

## Important Design Principle

The answer-generation prompt instructs GPT-4o-mini to use **only the retrieved Knowledge Graph Context**.

This reduces unsupported answers and makes the system a grounded RAG pipeline rather than a general-purpose chatbot.

## Current Limitations

The current implementation is a prototype.

### 1. Retrieval

The current retrieval approach primarily uses keyword/entity matching and one-hop relationships.

A production version should support:

- Multi-hop graph traversal
- Question intent detection
- Better entity linking
- Relationship-aware retrieval
- Hybrid graph + vector/text retrieval

### 2. Extraction Quality

Some extracted relationships may have incorrect direction or semantics.

For example, relationship direction needs to be validated so that:

```text
Students --VISITS--> Adani Kattupalli Port
```

is not incorrectly represented as:

```text
Adani Kattupalli Port --VISITS--> Students
```

### 3. Provenance

The next version should preserve document/chunk/page provenance for graph facts so answers can provide source references.

## Roadmap

### Phase 1 — Core correctness

- [x] Neo4j setup
- [x] PDF processing
- [x] Entity extraction
- [x] Relationship extraction
- [x] Graph construction
- [x] Basic graph retrieval
- [x] LLM answer generation
- [x] Interactive question analysis
- [ ] Improve relationship extraction quality
- [ ] Multi-hop graph retrieval
- [ ] Hybrid Graph + Text RAG

### Phase 2 — Application

- [ ] Conversational memory
- [ ] Source/page citations
- [ ] Streamlit web interface
- [ ] Graph evidence display
- [ ] Better error handling

### Phase 3 — Evaluation and Deployment

- [ ] Create evaluation questions
- [ ] Measure retrieval accuracy
- [ ] Measure answer accuracy
- [ ] Test hallucination resistance
- [ ] Package application
- [ ] Deploy

## Security

Do not commit:

```text
.env
venv/
API keys
Neo4j passwords
private documents
```

Before pushing to GitHub, verify:

```powershell
git status
```

and make sure `.env` is not listed as a file to be committed.

## Development Environment

Current project location:

```text
C:\Users\Lenovo\GenAI\HG-RAG
```

## License

Add an appropriate license before public distribution.
