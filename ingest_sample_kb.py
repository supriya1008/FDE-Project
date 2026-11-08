from app.services.ingestion import load_file,chunk_documents
from app.rag.vectorestore import add_documents,get_retriever
from pathlib import Path
from app.core.config import Settings


settings=Settings()

folder=Path(settings.sample_kb_dir)

files=[p for p in folder.iterdir() if p.is_file()]

all_docs=[]

for path in files:
    all_docs.extend(load_file(path))
chunks=chunk_documents(all_docs)
ids=add_documents(chunks)
print(f"Added {len(ids)} documents to the vector store.")
