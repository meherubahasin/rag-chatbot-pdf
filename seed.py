import os
from pathlib import Path
from agent import search_similar
from langchain_community.document_loaders import PyPDFLoader
from sqlalchemy import create_engine, text
from local_ollama import ask_ollama, get_embedding
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
import numpy as np
import pickle
import numpy as np 
import re 
from sklearn.metrics.pairwise import cosine_similarity 

CONNECTION_STRING = os.environ.get("DATABASE_URL", "postgresql+psycopg2://postgres:123456@localhost:5432/mydatabase")
engine = create_engine(CONNECTION_STRING)
COLLECTION_NAME = "ollama_pdf_docs"
bm25_docs = []
bm25_index = None

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader



# --- Semantic Chunking Helpers ---

def split_text_into_sentences(text):
    """Split text into sentences using regex."""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.?!])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def semantic_chunking_split(document: Document, threshold: float = 0.80, min_chunk_size: int = 300, overlap: int = 150):
    """
    Split document content into semantically coherent chunks.
    - threshold: cosine similarity below which a new chunk starts
    - min_chunk_size: minimum characters per chunk
    - overlap: number of characters to overlap between consecutive chunks
    """
    content = document.page_content
    sentences = split_text_into_sentences(content)
    if not sentences:
        return []

    # Get embeddings for each sentence
    embeddings = [get_embedding(s) for s in sentences]
    embeddings_matrix = np.array(embeddings)

    # Compute cosine similarity between adjacent sentences
    similarities = cosine_similarity(embeddings_matrix[:-1], embeddings_matrix[1:])
    similarities = similarities.diagonal()

    # Identify breakpoints for chunking
    breakpoints = [0]
    for i, sim in enumerate(similarities):
        if sim < threshold:
            breakpoints.append(i + 1)
    if breakpoints[-1] != len(sentences):
        breakpoints.append(len(sentences))

    # Build chunks with overlap
    chunks = []
    for i in range(len(breakpoints) - 1):
        start, end = breakpoints[i], breakpoints[i + 1]
        chunk_sentences = sentences[start:end]
        chunk_text = " ".join(chunk_sentences)

        # Apply overlap by including sentences from previous chunk
        if chunks and overlap > 0:
            prev_text = " ".join(sentences[max(0, start - overlap):start])
            chunk_text = prev_text + " " + chunk_text

        if len(chunk_text) >= min_chunk_size:
            chunks.append(Document(page_content=chunk_text, metadata=document.metadata))

    return chunks

# --- Database insert helper ---
def insert_chunk(content: str):
    emb = get_embedding(content)
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO pdf_embeddings (content, embedding) VALUES (:c, :e)"),
            {"c": content, "e": emb}
        )
def seed_database(pdf_dir_path="./pdfs"):
    """
    Process all PDFs in the folder, semantically chunk them,
    insert embeddings into the DB, and build BM25 index.
    """
    pdf_dir = Path(pdf_dir_path)
    if not pdf_dir.exists():
        print(f"The directory {pdf_dir_path} does not exist.")
        return

    for pdf_file in pdf_dir.glob("*.pdf"):
        seed_single_document(pdf_file)

    build_bm25_index()
    
# --- Seed single PDF ---
def seed_single_document(file_path):
    """
    Loads, semantically chunks, and embeds a single PDF document.
    Handles empty files, encoding errors, and connection issues gracefully.
    """
    print(f"📄 Processing: {file_path.name}")

    if os.path.getsize(file_path) == 0:
        print(f"Skipping empty PDF: {file_path.name}")
        return

    try:
        loader = PyPDFLoader(str(file_path))
        documents = loader.load()
    except Exception as e:
        print(f"Error loading {file_path.name}: {e}. Skipping.")
        return

    all_chunks = []
    for doc in documents:
        if not doc.page_content.strip():
            print(f"Skipping empty page in {file_path.name}")
            continue

        try:
            chunks = semantic_chunking_split(doc, threshold=0.8, min_chunk_size=300, overlap=250)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Chunking failed for {file_path.name}: {e}")

    # Remove duplicate chunks
    unique_chunks = list({c.page_content: c for c in all_chunks}.values())

    for i, chunk in enumerate(unique_chunks, 1):
        try:
            insert_chunk(chunk.page_content)
            bm25_docs.append(chunk)
            if i % 10 == 0:
                print(f"Inserted {i}/{len(unique_chunks)} chunks from {file_path.name}")
        except ConnectionError as ce:
            print(f"Skipping chunk due to Ollama connection issue: {ce}")
        except Exception as e:
            print(f"DB insert failed for {file_path.name} chunk {i}: {e}")

    print(f"🎯 Finished {file_path.name} — Total valid chunks: {len(unique_chunks)}")


# --- Build BM25 index after all PDFs ---
def build_bm25_index():
    global bm25_index
    if bm25_docs:
        bm25_index = BM25Retriever.from_texts(
            [doc.page_content for doc in bm25_docs],
            k=20,
            metadatas=[doc.metadata for doc in bm25_docs]
        )
        with open("bm25_index.pkl", "wb") as f:
            pickle.dump(bm25_index, f)
        print("BM25 index built and saved successfully.")
