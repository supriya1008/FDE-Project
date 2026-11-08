from app.services.ingestion import load_file, chunk_documents
from pathlib import Path
from app.rag.vectorestore import add_documents, get_retriever

docs= load_file(Path("data/sample_kb/company_it_handbook.md"))
chunks = chunk_documents(docs)
add_documents(chunks)

retriever = get_retriever()
retrieved_docs = retriever.invoke("What is the company's IT support policy?")
print(f"Retrieved {len(retrieved_docs)} documents:")