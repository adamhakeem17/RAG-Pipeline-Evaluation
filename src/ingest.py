import os
import pdfplumber

def extract_text(pdf_path):
    if pdf_path is None or pdf_path == "":
        raise ValueError("pdf_path must be a non-empty string.")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at path: {pdf_path}")
    text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text())
    return "\n".join(text)

def chunk_text(text, chunk_size=1024, overlap=200):
    # Validate chunking parameters to avoid infinite loops.
    if text is None:
        raise ValueError("text must be provided for chunking.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and less than chunk_size.")
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
