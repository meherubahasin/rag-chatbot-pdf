import ollama
import shutil
from pathlib import Path
from sqlalchemy import create_engine, text
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# -----------------------------
# Database connection
# -----------------------------
engine = create_engine("postgresql+psycopg2://postgres:123456@localhost:5432/mydb")

with engine.begin() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS pdf_embeddings (
            id SERIAL PRIMARY KEY,
            content TEXT,
            embedding vector(768)
        )
    """))

# -----------------------------
# Embeddings helper
# -----------------------------
def get_embedding(text: str):
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]

def to_pgvector(vec):
    return "[" + ",".join(str(x) for x in vec) + "]"

# -----------------------------
# Insert one chunk
# -----------------------------
def insert_chunk(content: str):
    emb = get_embedding(content)
    emb_str = to_pgvector(emb)
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO pdf_embeddings (content, embedding) VALUES (:c, :e)"),
            {"c": content, "e": emb_str}
        )

# -----------------------------
# Upload and process PDF
# -----------------------------
def process_pdf(pdf_path):
    pdf_dir = Path("./pdfs")
    pdf_dir.mkdir(exist_ok=True)

    dest_path = pdf_dir / Path(pdf_path).name
    shutil.copy(pdf_path, dest_path)
    print(f"📂 Copied {pdf_path} → {dest_path}")

    docs = PyPDFLoader(str(dest_path)).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    doc_splits = splitter.split_documents(docs)

    print(f"📄 Loaded {len(docs)} pages, split into {len(doc_splits)} chunks")
    for split in doc_splits:
        insert_chunk(split.page_content)

    print("✅ PDF indexed into Postgres")

# -----------------------------
# Query similar
# -----------------------------
def search_similar(query: str, k: int = 5):
    query_emb = get_embedding(query)
    query_emb_str = to_pgvector(query_emb)
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT content, embedding <-> :query_embedding AS distance
                FROM pdf_embeddings
                ORDER BY distance
                LIMIT :k;
            """),
            {"query_embedding": query_emb_str, "k": k}
        )
        return [row.content for row in result]

# -----------------------------
# Ask Llama3.1
# -----------------------------
def ask_with_context(question: str):
    docs = search_similar(question, k=8)
    if not docs:
        return "Not in the provided documents."

    context = "\n\n---\n\n".join(docs)[:6000]
    response = ollama.chat(
        model="llama3.1",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Only answer from context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )
    return response["message"]["content"].strip()

# -----------------------------
# Interactive loop
# -----------------------------
if __name__ == "__main__":
    pdf_path = input("📤 Enter path of PDF to upload: ").strip()
    process_pdf(pdf_path)

    print("\n✅ You can now ask questions. Type 'exit' to quit.\n")
    while True:
        q = input("❓ Question: ")
        if q.lower() in ["exit", "quit"]:
            break
        print("💡 Answer:", ask_with_context(q))
