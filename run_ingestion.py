from dotenv import load_dotenv
load_dotenv()

from src.ingestion.parse_and_chunk import parse_and_chunk_all
from src.ingestion.embed_and_index import get_qdrant_client, embed_and_upsert

if __name__ == "__main__":
    print("Parsing and chunking all collections...")
    records = parse_and_chunk_all()
    print(f"\nTotal chunks across all collections: {len(records)}")

    print("\nEmbedding and indexing into Qdrant...")
    client = get_qdrant_client()
    embed_and_upsert(records, client)

    print("\nDone. Check http://localhost:6333/dashboard")