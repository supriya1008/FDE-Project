from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import time
from app.core.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings



settings = Settings()
_embeddings = None
_vectorstore = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    return _embeddings

def get_embedding_dimension():
    embeddings = get_embeddings()
    return len(embeddings.embed_query("dimension check"))

from pinecone import Pinecone, ServerlessSpec
import time

def ensure_index():

    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing")

    pc = Pinecone(api_key=settings.pinecone_api_key)

    index_name = settings.pinecone_index_name

    if index_name not in pc.list_indexes().names():

        pc.create_index(
            name=index_name,
            dimension=get_embedding_dimension(),
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    return pc.Index(index_name)

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        index = ensure_index()
        _vectorstore = PineconeVectorStore(
            index=index,
            embedding=get_embeddings(),
            namespace=settings.pinecone_namespace,
        )
    return _vectorstore

def get_retriever():
    return get_vectorstore().as_retriever(search_kwargs={"k": settings.top_k})

def add_documents(chunks):
    store = get_vectorstore()
    return store.add_documents(chunks)