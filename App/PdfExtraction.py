import fitz
import re

class Pdf_Parser:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
    
    # Extraction of PDF
    def Text_Extraction(self):
        pages = []
        with fitz.open(self.pdf_path) as doc:
            for i, page in enumerate(doc, start=1):
                text = page.get_text()
                pages.append({"page": i, "text": text})
        return pages
    
    # Cleaning text
    def clean_text(self, text):
        # Replace carriage returns with newlines
        text = text.replace('\r', '\n')

        # Merge hyphenated words broken across lines
        text = re.sub(r'-\n', '', text)

        # Merge lines split within paragraphs
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

        # Remove multiple newlines
        text = re.sub(r'\n{2,}', '\n\n', text)

        # Remove excessive spaces/tabs
        text = re.sub(r'[ \t]{2,}', ' ', text)

        return text.strip()

if __name__ == "__main__":
    pdf_path = input("Enter the Path of your PDF: ").strip().strip('"').strip("'")
    obj = Pdf_Parser(pdf_path)

    # Extract text from PDF
    pages = obj.Text_Extraction()

    # Loop through pages and clean text
    for page in pages:
        page_num = page["page"]
        raw_text = page["text"]
        cleaned_text = obj.clean_text(raw_text)

        # Clearly show page number in output
        print(f"--- Page {page_num} ---")
        print(cleaned_text[:500])  # first 500 characters preview
        print("\n")
