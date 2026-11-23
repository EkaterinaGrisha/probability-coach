from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

from coach import config

COLLECTION = "coach"


def get_embeddings():
    # e5 обучали с префиксами "query: " и "passage: ", без них качество поиска заметно падает
    query, passage = ("query: ", "passage: ") if "e5" in config.EMBEDDING_MODEL else (None, None)
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True, "prompt": passage},
        query_encode_kwargs={"normalize_embeddings": True, "prompt": query},
    )


def get_store(collection=COLLECTION, **kwargs):
    url = config.DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)
    return PGVector(get_embeddings(), connection=url, collection_name=collection, **kwargs)
