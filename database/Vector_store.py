import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

class QdrantManager:
    def __init__(self, url: str, api_key: str):
        self.client = QdrantClient(url=url, api_key=api_key)

    def create_collection(self, collection_name: str, vector_size: int = 384):
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print(f"✅ Collection '{collection_name}' ready with vector size {vector_size}")

    def insert_embeddings(self, collection_name: str, chunks: list, embeddings: list):
        """
        Insert embeddings into collection, automatically generating UUIDs per chunk.
        Skips duplicates if the text already exists.
        """
        points_to_insert = []

        for i, chunk in enumerate(chunks):
            chunk_text = chunk["text"]
            
            # Generate deterministic UUID based on text
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_text))

            # Check if ID already exists in Qdrant
            try:
                existing = self.client.retrieve(collection_name=collection_name, ids=[chunk_id])
            except Exception:
                existing = None

            if existing and existing[0].id == chunk_id:
                # Skip duplicate
                continue

            points_to_insert.append({
                "id": chunk_id,
                "vector": embeddings[i],
                "payload": {
                    "text": chunk_text,
                    "source": chunk.get("source", "unknown"),
                    "page": chunk.get("page", None)
                }
            })

        if points_to_insert:
            self.client.upsert(collection_name=collection_name, points=points_to_insert)
            print(f"✅ Inserted {len(points_to_insert)} new embeddings into '{collection_name}'")
        else:
            print("⚠️ No new embeddings to insert (all duplicates)")

    def search(self, collection_name: str, query_vector: list, top_k: int = 3):
        """
        Search the collection for top-k most similar vectors
        """
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        return results
