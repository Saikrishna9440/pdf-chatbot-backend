import os
from App.Rag import RAGRetriever
from App.PdfExtraction import Pdf_Parser
import App.Chunkss as ch
import App.embeddings as ed
from database.Vector_store import QdrantManager
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# ✅ Load environment variablesa
load_dotenv()
qdrant_url = os.getenv("qdrant_url")
qdrant_apikey = os.getenv("qdrant_apikey")
gemini_key = os.getenv("Gemini_API_key")
genai.configure(api_key=gemini_key)

# ✅ Model initialization
model = genai.GenerativeModel("gemini-2.5-pro")

# ✅ Qdrant setup
collection_name = "Govt_scheme"
qdrant = QdrantManager(url=qdrant_url, api_key=qdrant_apikey)

# =================== 📄 PDF Upload + Indexing ===================
def upload_pdf():
    pdf_path = input("Enter the Path of your PDF: ").strip().strip('"').strip("'")
    parser = Pdf_Parser(pdf_path)
    pages = parser.Text_Extraction()

    # Clean the text
    for page in pages:
        page["text"] = parser.clean_text(page["text"])

    # Chunk the text
    chunks = ch.chunk_text(pages, chunk_size=300, overlap=30)
    print(f"✅ Total Chunks: {len(chunks)}")

    # Generate embeddings
    embeddings = ed.create_embeddings(chunks)
    print("✅ Embeddings generated")

    # Create collection if not exists
    vector_size = len(embeddings[0])
    qdrant.create_collection(collection_name, vector_size)

    # Insert into Qdrant
    qdrant.insert_embeddings(collection_name, chunks, embeddings)
    print("✅ PDF content successfully uploaded to Qdrant!")


# =================== 💬 Ask Questions ===================
def ask(query: str):
    retriever = RAGRetriever(qdrant_url, qdrant_apikey, collection_name)
    chunks = retriever.retrieve(query)

    if not chunks:
        return "I don't know"

    context = "\n\n".join([c["text"] for c in chunks if c.get("text")])

    prompt = f"""
    You are an expert on Indian Government Schemes.
    Answer the question ONLY using the provided context.
    If the answer is not explicitly in the context, reply exactly with: "I don't know".

    Question:
    {query}

    Context:
    {context}
    """

    response = model.generate_content(prompt)
    return response.text


# =================== 🚀 Main Program ===================
if __name__ == "__main__":
    print("=== Government Schemes Chatbot ===")
    choice = input("Do you want to upload a new PDF? (yes/no): ").strip().lower()

    if choice in ["yes", "y"]:
        upload_pdf()

    while True:
        query = input("\nAsk a question (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("👋 Goodbye!")
            break

        answer = ask(query)
        print("\n🤖 Answer:\n", answer)
