
import os
import re
from pypdf import PdfReader
from docx import Document
import io

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
    return text

def extract_text_from_docx(docx_path):
    text = ""
    try:
        document = Document(docx_path)
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        print(f"Error extracting text from DOCX {docx_path}: {e}")
    return text

def extract_text_from_txt(txt_path):
    text = ""
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"Error extracting text from TXT {txt_path}: {e}")
    return text

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    chunks = []
    # Split text by common delimiters to maintain sentence integrity
    # This is a basic approach, more advanced methods use NLTK/SpaCy
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length <= chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append(" ".join(current_chunk))
            # Add overlap
            # Simple overlap for demonstration, can be improved
            overlap_words = int(len(current_chunk) * chunk_overlap / chunk_size)
            overlap_text = " ".join(current_chunk[-overlap_words:])
            current_chunk = [overlap_text, sentence]
            current_length = len(overlap_text.split()) + sentence_length
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def process_document(file_path, chunk_size=1000, chunk_overlap=200):
    file_extension = os.path.splitext(file_path)[1].lower()
    extracted_text = ""

    if file_extension == ".pdf":
        extracted_text = extract_text_from_pdf(file_path)
    elif file_extension == ".docx":
        extracted_text = extract_text_from_docx(file_path)
    elif file_extension == ".txt":
        extracted_text = extract_text_from_txt(file_path)
    else:
        print(f"Unsupported file type: {file_extension}")
        return []

    if extracted_text:
        return chunk_text(extracted_text, chunk_size, chunk_overlap)
    return []

if __name__ == "__main__":
    # Example Usage (replace with actual file paths)
    # Create dummy files for testing
    with open("example.txt", "w") as f:
        f.write("This is a sample text file. It contains multiple sentences. We will try to chunk this text. This is the fourth sentence.")
    
    # For PDF and DOCX, you would need actual files.
    # For demonstration, we'll just use the TXT example.
    
    print("\n--- Processing example.txt ---")
    chunks = process_document("example.txt", chunk_size=50, chunk_overlap=10)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:\n{chunk}\n---")

    # Clean up dummy file
    os.remove("example.txt")



