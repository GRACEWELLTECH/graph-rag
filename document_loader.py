from langchain_community.document_loaders import PyPDFLoader

PDF_PATH = "documents/sample.pdf"

loader = PyPDFLoader(PDF_PATH)

documents = loader.load()

print(f"✅ PDF loaded successfully!")
print(f"Number of pages: {len(documents)}")

for i, doc in enumerate(documents[:3]):
    print(f"\n--- Page {i + 1} ---")
    print(doc.page_content[:1000])