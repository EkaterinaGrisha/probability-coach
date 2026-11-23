import logging
from functools import cache

from langsmith import traceable
from sentence_transformers import CrossEncoder

from coach import config
from coach.store import COLLECTION, get_store

CANDIDATES = 20
log = logging.getLogger("coach")


def _trace_documents(docs):
    return {"documents": [{"page_content": d.page_content, "type": "Document", "metadata": d.metadata} for d in docs]}


@cache
def _store(collection):
    return get_store(collection)


@cache
def _reranker():
    return CrossEncoder(config.RERANKER_MODEL)


@traceable(run_type="retriever", name="pgvector", process_outputs=_trace_documents)
def vector_search(query, k, flt, collection=COLLECTION):
    found = _store(collection).similarity_search_with_score(query, k=k, filter=flt)
    for doc, distance in found:
        doc.metadata["score"] = round(1 - distance, 3)
    return [doc for doc, _ in found]


@traceable(run_type="retriever", name="rerank", process_outputs=_trace_documents)
def rerank(query, docs, k):
    if not docs:
        return []
    scores = _reranker().predict([(query, d.page_content) for d in docs])
    ranked = sorted(zip(scores, docs), key=lambda pair: pair[0], reverse=True)[:k]
    for score, doc in ranked:
        doc.metadata["rerank_score"] = round(float(score), 3)
    return [doc for _, doc in ranked]


def search(query, k, flt, collection=COLLECTION, use_reranker=bool(config.RERANKER_MODEL), question=None):
    if not use_reranker:
        docs = vector_search(query, k, flt, collection)
    else:
        docs = rerank(query, vector_search(query, CANDIDATES, flt, collection), k)
    if use_reranker and question and question != query:
        # запрос агента теряет часть смысла вопроса, поэтому ищем и по реплике пользователя. Каждого кандидата
        # оцениваем тем запросом, который его нашёл: реплика вроде «50 на 50, уверена на 85%» получает низкие
        # оценки и не вытесняет найденное по запросу агента
        docs += rerank(question, vector_search(question, CANDIDATES, flt, collection), k)
        best = {}
        for doc in sorted(docs, key=lambda d: d.metadata["rerank_score"], reverse=True):
            best.setdefault(doc.metadata["chunk_id"], doc)
        docs = list(best.values())[:k]
    for doc in docs:
        log.info("CHUNK %s score=%s rerank=%s", doc.metadata["chunk_id"], doc.metadata["score"],
                 doc.metadata.get("rerank_score", "-"))
    return docs
