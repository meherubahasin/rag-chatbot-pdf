import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sqlalchemy import create_engine, text
from local_ollama import ask_ollama, get_embedding
from langchain_community.retrievers import BM25Retriever
import pickle

CONNECTION_STRING = "postgresql+psycopg2://postgres:123456@localhost:5432/mydatabase"
engine = create_engine(CONNECTION_STRING)
COLLECTION_NAME = "ollama_pdf_docs"
bm25_docs = []
bm25_index = None

def insert_chunk(content: str):
    try:
        emb = get_embedding(content)
        with engine.begin() as conn:
            conn.execute(
                text("INSERT INTO pdf_embeddings (content, embedding) VALUES (:c, :e)"),
                {"c": content, "e": emb}
            )
    except ConnectionError as e:
        print(f"Error inserting chunk: {e}")
        raise

def seed_single_document(file_path):
    try:
        loader = PyPDFLoader(str(file_path))
        documents = loader.load()
    except Exception as e:
        print(f"Error loading {file_path.name}: {e}. Skipping this file.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    doc_splits = text_splitter.split_documents(documents)

    global bm25_docs
    bm25_docs.extend(doc_splits)

    for i, split in enumerate(doc_splits, 1):
        try:
            insert_chunk(split.page_content)
        except Exception as e:
            print(f"Error inserting chunk {i}: {e}")
            continue

def seed_database():
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS pdf_embeddings"))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS pdf_embeddings (
                id SERIAL PRIMARY KEY,
                content TEXT,
                embedding vector(768)
            )
        """))
    
    pdf_dir = Path("./pdfs")
    
    if not pdf_dir.exists():
        print("The 'pdfs' directory does not exist. Please create it and add your PDF files.")
        return
        
    for pdf in pdf_dir.glob("*.pdf"):
        seed_single_document(pdf)
    
    # Build BM25 index
    if bm25_docs:
        bm25_index = BM25Retriever.from_texts([doc.page_content for doc in bm25_docs], k=20)
        with open("bm25_index.pkl", "wb") as f:
            pickle.dump(bm25_index, f)

def search_similar(query: str, k: int = 5):
    try:
        # Generate hypothetical answer for HyDE
        hyde_prompt = f"Provide a concise hypothetical answer to the following question: {query}"
        hypothetical_answer = ask_ollama("", hyde_prompt)
        
        # Use hypothetical answer for embedding
        query_emb = get_embedding(hypothetical_answer['answer'])

        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT id, content, embedding <-> (:query_embedding)::vector AS distance
                    FROM pdf_embeddings
                    ORDER BY distance
                    LIMIT :k;
                """),
                {"query_embedding": query_emb, "k": k}
            )
            vector_results = [{"id": row.id, "content": row.content, "score": 1 - row.distance} for row in result]
        
        # BM25 search (using original query)
        if os.path.exists("bm25_index.pkl"):
            with open("bm25_index.pkl", "rb") as f:
                bm25_index = pickle.load(f)
            bm25_results = bm25_index.get_relevant_documents(query)
            # Reformat BM25 results
            bm25_results = [doc.page_content for doc in bm25_results]
        else:
            bm25_results = []
            
        combined_docs = list(set([res['content'] for res in vector_results] + bm25_results))

        return combined_docs
    except Exception as e:
        print(f"Database query or search failed: {e}")
        return []

def ask_with_context(question: str):
    docs = search_similar(question, k=20)
    
    if not docs:
        return "Not in the provided documents."
    
    context = "\n\n".join(docs)
    return ask_ollama(context, question)