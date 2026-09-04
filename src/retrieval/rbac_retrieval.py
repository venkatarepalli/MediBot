"""
Phase 3: RBAC-filtered retrieval.

Enforces access control at the Qdrant query layer itself — restricted chunks
are excluded from the candidate set before any embedding similarity is even
computed. This means the LLM never sees content outside the caller's role,
so no prompt (adversarial or otherwise) can make it leak restricted content.
"""
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding

from src.rbac.access_config import validate_role
from src.ingestion.embed_and_index import QDRANT_COLLECTION, DENSE_MODEL

_dense_model = None  # lazy-loaded singleton, avoid reloading on every call


def _get_dense_model() -> TextEmbedding:
    global _dense_model
    if _dense_model is None:
        _dense_model = TextEmbedding(model_name=DENSE_MODEL)
    return _dense_model


def rbac_search(client: QdrantClient, role: str, query_text: str, limit: int = 10) -> list[dict]:
    """
    Dense-only RBAC-filtered search (hybrid fusion + reranking added in Phase 4).

    Returns a list of dicts: {text, source_document, section_title, collection, chunk_type, score}
    Only chunks whose access_roles list contains `role` can appear in the results.
    """
    validate_role(role)

    query_vector = next(_get_dense_model().embed([query_text])).tolist()

    result = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        using="dense",
        query_filter=models.Filter(
            must=[models.FieldCondition(key="access_roles", match=models.MatchValue(value=role))]
        ),
        limit=limit,
        with_payload=True,
    )

    return [
        {
            "text": p.payload["text"],
            "source_document": p.payload["source_document"],
            "section_title": p.payload["section_title"],
            "collection": p.payload["collection"],
            "chunk_type": p.payload["chunk_type"],
            "score": p.score,
        }
        for p in result.points
    ]