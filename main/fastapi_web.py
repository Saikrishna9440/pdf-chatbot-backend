# app/main.py
import sys
import os

# Make sure the parent directory of 'App' is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import shutil
from App.Rag import RAGRetriever
from App.PdfExtraction import Pdf_Parser
from App.Chunkss import Chunking
from App.embeddings import Embed
from database.Vector_store import QdrantManager
import google.generativeai as genai
from dotenv import load_dotenv

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
        # Save uploaded PDF temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract and clean text
        parser = Pdf_Parser(temp_path)
        pages = parser.Text_Extraction()
        for page in pages:
            page["text"] = parser.clean_text(page["text"])

        # Chunk the text
        ch = Chunking(pages,300,30)
        chunks = ch.chunk_text()

        # Generate embeddings
        ed = Embed(chunks)
        embeddings = ed.create_embeddings()

        # Create collection if not exists
        vector_size = len(embeddings[0])
        qdrant.create_collection(collection_name, vector_size)

        # Insert into Qdrant
        qdrant.insert_embeddings(collection_name, chunks, embeddings)

        # Remove temporary file
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

        if not chunks:
            return {"answer": "I don't know"}

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
        return {"answer": response.text}

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
"""uvicorn is used to run the asgi server 
syntax: uvicorn filename:directoryname --reload"""