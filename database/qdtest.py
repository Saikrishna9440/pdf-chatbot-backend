from App.PdfExtraction import Pdf_Parser
import App.Chunkss as ch
import App.embeddings as ed
from database.Vector_store import QdrantManager
from sentence_transformers import SentenceTransformer, util
import os
from dotenv import load_dotenv

QDRANT_URL = os.getenv("qdrant_url")
QDRANT_API_KEY = os.getenv("qdrant_apikey")

if __name__ == "__main__":
    pdf_path = input("Enter the Path of your PDF: ").strip().strip('"').strip("'")
    parser = Pdf_Parser(pdf_path)
    pages = parser.Text_Extraction()

    # clean text
    for page in pages:
        page["text"] = parser.clean_text(page["text"])

    #chunks
    chunks = ch.chunk_text(pages, chunk_size=300, overlap=30)
    print(f"✅ Total Chunks: {len(chunks)}")

    #embeedings
    embeddings = ed.create_embeddings(chunks)
    print("✅ Embeddings generated")

    #connecting qdrant
    qdrant = QdrantManager(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    collection_name = "Govt_scheme"
    vector_size = len(embeddings[0])
    qdrant.create_collection(collection_name, vector_size)
    
    qdrant.insert_embeddings(collection_name, chunks, embeddings)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    user_query = input("Ask a question about your PDF: ").strip()
    query_vector = model.encode([user_query])[0]

    results = qdrant.search(collection_name, query_vector, top_k=5)

    print("\n🔎 Top results:")
    for r in results:
        print(f"Score: {r.score:.4f} | Text: {r.payload['text'][:300]}...\n")




    



