from sentence_transformers import SentenceTransformer, util
from App.PdfExtraction import Pdf_Parser
from App.Chunkss import Chunking

class Embed:
    def __init__(self, chunks, model_name='all-MiniLM-L6-v2'):
        self.chunks = chunks
        self.model_name = model_name

    def create_embeddings(self):
        # Load model
        model = SentenceTransformer(self.model_name)

        # Filter out empty chunks
        texts = [chunk['text'] for chunk in self.chunks if chunk['text'].strip()]

        # Generate embeddings
        embeddings = model.encode(texts, show_progress_bar=True)

        # Return embeddings only (remove model from return to avoid tuple confusion)
        return embeddings


if __name__ == "__main__":
    pdf_path = input("Enter the Path of your PDF: ").strip().strip('"').strip("'")
    parser = Pdf_Parser(pdf_path)

    # Extract and clean PDF text
    pages = parser.Text_Extraction()
    for page in pages:
        page["text"] = parser.clean_text(page["text"])
    
    # Chunk text
    ch = Chunking(pages, 100, 20)
    chunks = ch.chunk_text()

    # Generate embeddings
    ed = Embed(chunks)
    embeddings = ed.create_embeddings()

    # Display embeddings info
    for i, emb in enumerate(embeddings[:3]):  # first 3 chunks for brevity
        print(f"Chunk {i+1} embedding dimension: {len(emb)}")
        print(emb[:10], "...\n")
