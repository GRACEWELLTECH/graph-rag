from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = "documents/sample.pdf"

# 1. Load PDF
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print(f"Pages loaded: {len(documents)}")

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Chunks created: {len(chunks)}")

# 3. Display first 3 chunks
for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- CHUNK {i + 1} ---")
    print(chunk.page_content[:1000])
    print(f"\nMetadata: {chunk.metadata}")