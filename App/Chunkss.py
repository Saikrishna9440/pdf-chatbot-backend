import re
from App.PdfExtraction import Pdf_Parser


class Chunking:
    def __init__(self, pages, chunk_size=200, overlap=20):
        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk_size.")
        self.pages = pages
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self):
        chunks = []

        for page in self.pages:
            text = page["text"]
            page_number = page["page"]
            source = page.get("source", "PDF")

            words = text.split()
            start = 0

            while start < len(words):
                end = min(start + self.chunk_size, len(words))
                chunk_words = words[start:end]
                chunk_text_str = " ".join(chunk_words)

                chunks.append({
                    "text": chunk_text_str,
                    "page": page_number,
                    "source": source
                })

                if end == len(words):
                    break
                start = end - self.overlap

        return chunks


if __name__ == "__main__":
    pdf_path = input("Enter the Path of your PDF: ").strip().strip('"').strip("'")
    
    # Create Pdf_Parser object
    parser = Pdf_Parser(pdf_path)
    
    # Extract pages
    pages = parser.Text_Extraction()
    
    # Clean text of each page
    for page in pages:
        page["text"] = parser.clean_text(page["text"])
    
    # Split into chunks
    ch = Chunking(pages, chunk_size=100, overlap=20)
    chunks = ch.chunk_text()

    # Preview first 5 chunks
    for i, chunk in enumerate(chunks[:5]):
        print(f"--- Chunk {i+1} ---")
        print(f"Page: {chunk['page']}, Source: {chunk['source']}")
        print(chunk["text"])
        print("\n")
