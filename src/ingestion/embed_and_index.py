"""
Embed every chunk with BOTH a dense vector (semantic) and a sparse BM25 vector,
and upsert into a single Qdrant collection using named vectors — so one
query_points() call can fuse both at retrieval time (server-side, not in app code).
"""
import uuid
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding

QDRANT_COLLECTION = "medibot_chunks"
DENSE_MODEL = "BAAI/bge-small-en-v1.5"   # 384-dim, good CPU speed/quality tradeoff
SPARSE_MODEL = "Qdrant/bm25"              # literal BM25 sparse vectors


def get_qdrant_client(url: str = "http://localhost:6333") -> QdrantClient:
    return QdrantClient(url=url)


def ensure_collection(client: QdrantClient, dense_dim: int = 384):
    if client.collection_exists(QDRANT_COLLECTION):
        return
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config={"dense": models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )


def embed_and_upsert(records: list[dict], client: QdrantClient, batch_size: int = 32):
    dense_model = TextEmbedding(model_name=DENSE_MODEL)
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    ensure_collection(client)

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        texts = [r["text"] for r in batch]
        dense_vecs = list(dense_model.embed(texts))
        sparse_vecs = list(sparse_model.embed(texts))

        points = []
        for record, dense_vec, sparse_vec in zip(batch, dense_vecs, sparse_vecs):
            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_vec.tolist(),
                    "sparse": models.SparseVector(
                        indices=sparse_vec.indices.tolist(),
                        values=sparse_vec.values.tolist(),
                    ),
                },
                payload={
                    "text": record["text"],  # contextualized (heading-aware) — reranker + LLM need this, not raw_text
                    "source_document": record["source_document"],
                    "collection": record["collection"],
                    "access_roles": record["access_roles"],
                    "section_title": record["section_title"],
                    "chunk_type": record["chunk_type"],
                },
            ))
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        print(f"  upserted {i + len(batch)}/{len(records)}")