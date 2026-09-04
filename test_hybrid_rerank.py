from dotenv import load_dotenv
load_dotenv()

from src.ingestion.embed_and_index import get_qdrant_client
from src.retrieval.hybrid_rerank import hybrid_search, rerank

if __name__ == "__main__":
    client = get_qdrant_client()
    role, query = "nurse", "What is the correct IV cannula size for a paediatric patient under 5kg?"

    candidates = hybrid_search(client, role, query, prefetch_limit=10)
    print(f"Hybrid candidates (top-10, pre-rerank):")
    for c in candidates:
        print(f"  [{c['collection']}] {c['source_document']} :: {c['section_title']}")

    top = rerank(query, candidates, top_n=3)
    print(f"\nAfter reranking (top-3):")
    for t in top:
        print(f"  score={t['rerank_score']:.3f}  [{t['collection']}] {t['source_document']} :: {t['section_title']}")