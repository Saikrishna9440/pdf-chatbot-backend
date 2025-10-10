# app/main.py
import sys
import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import google.generativeai as genai
from dotenv import load_dotenv

# Make sure the parent directory of 'App' is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Local imports
from App.Rag import RAGRetriever
from App.PdfExtraction import Pdf_Parser
from App.Chunkss import Chunking
from App.embeddings import Embed
from database.Vector_store import QdrantManager
from main.newsapi import fetch_GovtScheme_news  # ✅ Import your async News API function

load_dotenv()

# =================== 🔑 Environment Setup ===================
qdrant_url = os.getenv("qdrant_url")
qdrant_apikey = os.getenv("qdrant_apikey")
gemini_key = os.getenv("Gemini_API_key")
genai.configure(api_key=gemini_key)

# =================== ✅ Model & Qdrant ===================
model = genai.GenerativeModel("gemini-2.5-pro")
collection_name = "Govt_scheme"
qdrant = QdrantManager(url=qdrant_url, api_key=qdrant_apikey)

# =================== 🚀 FastAPI App ===================
app = FastAPI(title="Government Schemes Chatbot")

# =================== 📄 PDF Upload Endpoint ===================
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        parser = Pdf_Parser(temp_path)
        pages = parser.Text_Extraction()
        for page in pages:
            page["text"] = parser.clean_text(page["text"])

        ch = Chunking(pages, 300, 30)
        chunks = ch.chunk_text()

        ed = Embed(chunks)
        embeddings = ed.create_embeddings()

        vector_size = len(embeddings[0])
        qdrant.create_collection(collection_name, vector_size)
        qdrant.insert_embeddings(collection_name, chunks, embeddings)

        os.remove(temp_path)
        return JSONResponse({"status": "success", "total_chunks": len(chunks)})

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# =================== 💬 Ask Question Endpoint ===================
@app.post("/ask")
async def ask_question(query: str = Form(...)):
    try:
        retriever = RAGRetriever(qdrant_url, qdrant_apikey, collection_name)
        chunks = retriever.retrieve(query)

        relevant_chunks = [c for c in chunks if c.get("score", 0) > 0.55]

        context_text = ""
        if relevant_chunks:
            context_text = "\n\n".join([c["text"] for c in relevant_chunks if c.get("text")])

        if not context_text:
            news_data = fetch_GovtScheme_news(query)
            if "news" in news_data:
                news_texts = [
                    f"Title: {n['title']}\nDescription: {n['description']}\nSource: {n['source']}\nURL: {n['url']}"
                    for n in news_data["news"]
                ]
                context_text = "\n\n".join(news_texts)
            else:
                return {"answer": "I don't know."}

        with open("Prompts/system_prompt", "r", encoding="utf-8") as f:
            sys_prompt = f.read()

        prompt = f"""
        {sys_prompt}

        Question:
        {query}

        Context:
        {context_text}
        """

        response = model.generate_content(prompt)
        return {"answer": response.text}

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


"""
💡 To run:
uvicorn main:app --reload
"""
