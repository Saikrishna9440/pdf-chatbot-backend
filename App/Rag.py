from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

class RAGRetriever:
    def __init__(self, qdrant_url: str, api_key: str = None, collection_name: str = None, embedding_model: str = "all-MiniLM-L6-v2"):
        self.client = QdrantClient(url=qdrant_url, api_key=api_key)
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer(embedding_model)

    def retrieve(self, query: str, top_k: int = 3):
        """
        Retrieve top-k most similar chunks for a given query.
        """
        query_vector = self.embedding_model.encode(query).tolist()
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )

        retrieved_chunks = []
        for res in results:
            payload = res.payload or {}
            retrieved_chunks.append({
                "text": payload.get("text", ""),
                "page": payload.get("page", None),
                "source": payload.get("source", "unknown"),
                "score": float(res.score) if hasattr(res, "score") else None
            })
        return retrieved_chunks
