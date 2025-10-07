import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from App.Rag import RAGRetriever
from App.PdfExtraction import Pdf_Parser
from  App.Chunkss import Chunking
from  App.embeddings import Embed
from database.Vector_store import QdrantManager
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
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
    ch = Chunking(pages,300,30)
    chunks = ch.chunk_text()
    print(f"✅ Total Chunks: {len(chunks)}")

    # Generate embeddings
    ed=Embed(chunks)
    embeddings = ed.create_embeddings()
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
    file_path = "Prompts/system_prompt"
    with open(file_path,"r") as f:
        sys_prompt=f.read()

    prompt = f"""
    {sys_prompt}

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
        query = input("\nAsk a question (or type 'exit' to quiyet): ").strip()
        if query.lower() == "exit":
            print("👋 Goodbye!")
            break

        answer = ask(query)
        print("\n🤖 Answer:\n", answer)
