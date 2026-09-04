"""
Phase 4: Hybrid retrieval (dense + BM25 fusion) + cross-encoder reranking,
with RBAC enforced at the Qdrant query layer.

IMPORTANT: the RBAC filter is applied to EACH prefetch leg individually, not
just as a top-level query_filter — Qdrant does not apply a top-level filter
across prefetch+fusion queries (verified empirically). Relying on the
top-level filter alone would silently leak restricted content.
"""
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from src.rbac.access_config import validate_role
from src.ingestion.embed_and_index import QDRANT_COLLECTION, DENSE_MODEL, SPARSE_MODEL

RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

_dense_model = None
_sparse_model = None
_reranker = None


def _get_dense_model() -> TextEmbedding:
    global _dense_model
    if _dense_model is None:
        _dense_model = TextEmbedding(model_name=DENSE_MODEL)
    return _dense_model


def _get_sparse_model() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    return _sparse_model


def _get_reranker() -> TextCrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = TextCrossEncoder(model_name=RERANK_MODEL)
    return _reranker


def hybrid_search(client: QdrantClient, role: str, query_text: str, prefetch_limit: int = 10) -> list[dict]:
    """Broad candidate retrieval: dense + BM25 sparse, fused with RRF. RBAC filter on BOTH legs."""
    validate_role(role)

    dense_vec = next(_get_dense_model().embed([query_text])).tolist()
    sparse_vec = next(_get_sparse_model().embed([query_text]))

    role_filter = models.Filter(
        must=[models.FieldCondition(key="access_roles", match=models.MatchValue(value=role))]
    )

    result = client.query_points(
        collection_name=QDRANT_COLLECTION,
        prefetch=[
            models.Prefetch(query=dense_vec, using="dense", limit=prefetch_limit, filter=role_filter),
            models.Prefetch(
                query=models.SparseVector(indices=sparse_vec.indices.tolist(), values=sparse_vec.values.tolist()),
                using="sparse", limit=prefetch_limit, filter=role_filter,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=role_filter,  # redundant given the leg filters, kept as defense-in-depth
        limit=prefetch_limit,
        with_payload=True,
    )

    return [
        {
            "text": p.payload["text"],
            "source_document": p.payload["source_document"],
            "section_title": p.payload["section_title"],
            "collection": p.payload["collection"],
            "chunk_type": p.payload["chunk_type"],
        }
        for p in result.points
    ]


def rerank(query_text: str, candidates: list[dict], top_n: int = 3) -> list[dict]:
    """Cross-encoder reranking: scores query+candidate jointly, keeps only top_n."""
    if not candidates:
        return []
    documents = [c["text"] for c in candidates]
    scores = list(_get_reranker().rerank(query_text, documents))
    scored = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [{**candidate, "rerank_score": float(score)} for candidate, score in scored[:top_n]]


def hybrid_rerank_search(client: QdrantClient, role: str, query_text: str,
                          prefetch_limit: int = 10, top_n: int = 3) -> list[dict]:
    """Full Phase 4 pipeline: hybrid retrieval -> rerank -> top_n."""
    candidates = hybrid_search(client, role, query_text, prefetch_limit=prefetch_limit)
    return rerank(query_text, candidates, top_n=top_n)