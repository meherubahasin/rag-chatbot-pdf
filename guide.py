# %% [markdown]
# RAG Chatbot with LangChain + Ollama + PostgreSQL (pgvector)


# %% [markdown]
## 1) Install required packages

# Run these commands in a terminal or a Jupyter cell (prefixed with `!` in notebooks).

# Core python packages
# !pip install -U langchain langchain-postgres langchain-ollama langchain-ollama==0.0.0  # (version pin optional)
# Embeddings + utils
# !pip install -U sentence-transformers transformers torch
# Postgres driver + pgvector python bindings
# !pip install -U psycopg[binary] pgvector sqlalchemy
# Optional: python-dotenv for env var loading
# !pip install -U python-dotenv

# NOTE: If you prefer OpenAI embeddings, install openai and relevant langchain helper packages.

# %% [markdown]
## 2) Start Ollama and pull a model (shell)

# Steps (run in terminal, not inside Python):
# 1. Install Ollama from https://ollama.ai (download + follow platform install instructions).
# 2. Start the server (headless) if you don't want GUI:
#    ollama serve
# 3. Pull a model (example: gemma:2b or llama3.2 if available locally/authorized):
#    ollama pull gemma:2b
# 4. Run a model (this keeps it loaded for CLI testing):
#    ollama run gemma:2b

# Alternatively you can interact via Ollama's REST API at http://localhost:11434 (default)

# Security note: do NOT expose the Ollama REST API to the public without authentication and network protections.

# %% [markdown]
## 3) Start Postgres with pgvector (Docker)

# Example Docker command to spin up a Postgres instance with `pgvector` preinstalled:
# docker run --name pgvector-container -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 5432:5432 -d pgvector/pgvector:pg16

# After the container starts, you can connect with psql or a GUI and enable the extension (if not enabled):
# psql -h localhost -U langchain -d langchain -p 5432
# Then inside psql:
# CREATE EXTENSION IF NOT EXISTS vector;

# %% [markdown]
## 4) Notebook: Imports & configuration

# %%
from dotenv import load_dotenv
import os
load_dotenv()

# Example env vars we might use (create a .env file or export them):
# OLLAMA_HOST (optional) e.g., "http://localhost:11434"
# PG_CONNECTION_STRING e.g., "postgresql+psycopg://langchain:langchain@localhost:5432/langchain"

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
PG_CONNECTION_STRING = os.getenv("PG_CONNECTION_STRING", "postgresql+psycopg://langchain:langchain@localhost:5432/langchain")

print("Ollama host:", OLLAMA_HOST)
print("PG connection:", PG_CONNECTION_STRING)

# %% [markdown]
## 5) Load LangChain & embedding tools

# %%
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Embeddings: use HuggingFace sentence-transformers (local, fast)
from langchain.embeddings import HuggingFaceEmbeddings

# PGVector vectorstore
from langchain_postgres.vectorstores import PGVector

# Retrieval QA chain
from langchain.chains import RetrievalQA

# Ollama integration for LangChain
# There are small community packages/wrappers; LangChain docs show using `langchain_ollama` or `langchain-ollama`
try:
    # preferred chat wrapper
    from langchain_ollama import ChatOllama
except Exception:
    # fallback: direct low-level LLM wrapper
    try:
        from langchain_ollama.llms import OllamaLLM as OllamaLLM
    except Exception:
        OllamaLLM = None
        ChatOllama = None

print("Loaded wrappers. ChatOllama available:", 'ChatOllama' in globals())

# %% [markdown]
## 6) Data ingestion, parsing & chunking

# Replace this with your own loader function: PDF, txt, markdown, web, etc.
# We'll show a simple example for local text files and a PDF example using pypdf (optional)

# Example: load multiple text files from a folder into LangChain Documents

# %%
from pathlib import Path

def load_texts_from_folder(folder: str, ext: str = "*.txt"):
    p = Path(folder)
    docs = []
    for f in p.glob(ext):
        texts = f.read_text(encoding='utf-8')
        docs.append(Document(page_content=texts, metadata={"source": str(f)}))
    return docs

# Small demo: create a few documents
example_docs = [Document(page_content="This is a short doc about LangChain and RAG." , metadata={"source":"demo1"}),
                Document(page_content="Ollama runs LLMs locally and exposes a REST API. Great for privacy." , metadata={"source":"demo2"})]

# Chunk documents into smaller pieces — good for embeddings and retrieval
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunked_docs = splitter.split_documents(example_docs)
len(chunked_docs), chunked_docs[0]

# %% [markdown]
## 7) Create embeddings for chunks

# NOTE: Choose a model compatible with HuggingFace Embeddings wrapper. `all-MiniLM-L6-v2` is a good default.

# %%
embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)

# Compute embeddings for one chunk (demo) or for all
texts = [d.page_content for d in chunked_docs]
# You can compute in batch
text_embeddings = embeddings.embed_documents(texts)
len(text_embeddings), len(text_embeddings[0])

# %% [markdown]
## 8) Persist embeddings into Postgres (PGVector)

# Use langchain-postgres PGVector wrapper — it will create tables automatically for the collection name.

# %%
COLLECTION_NAME = "my_docs_collection"

# Create the PGVector store from documents (this will run SQL to make tables and insert vectors)
# NOTE: Some PGVector implementations require a connection object; langchain-postgres expects connection strings.

try:
    vectordb = PGVector.from_documents(
        embedding=embeddings,
        documents=chunked_docs,
        collection_name=COLLECTION_NAME,
        connection_string=PG_CONNECTION_STRING,
        pre_delete_collection=True,
    )
    print("Inserted documents into PGVector collection")
except Exception as e:
    print("Error creating PGVector store:", e)
    print("You may need to install langchain-postgres and ensure Postgres + pgvector is running.")

# %% [markdown]
## 9) Build the Retriever + RAG chain using Ollama LLM

# We'll create a retriever from the PGVector store and pass it into a RetrievalQA chain wired to Ollama.

# %%
# Create a retriever
retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10})

# LLM: prefer Chat wrapper when dealing with conversation. Fallback to OllamaLLM if chat wrapper missing.
if 'ChatOllama' in globals() and ChatOllama is not None:
    llm = ChatOllama(model="gemma:2b", host=OLLAMA_HOST)
elif OllamaLLM is not None:
    llm = OllamaLLM(model="gemma:2b", host=OLLAMA_HOST)
else:
    raise RuntimeError("Ollama wrapper not available. Install `langchain-ollama` or `langchain_ollama` package.")

# Create RetrievalQA — you can choose chain_type 'stuff', 'map_reduce', or 'refine'
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)

# %% [markdown]
## 10) Simple chatbot REPL using the RAG chain

# This is a minimal loop that keeps the retrieval up-to-date and asks the LLM for an answer using retrieved context.

# %%

def chat_repl():
    print("RAG Chatbot (type 'exit' to quit)")
    while True:
        query = input("You: ")
        if not query or query.strip().lower() in ("exit", "quit"):
            break
        result = qa_chain(query)
        answer = result.get('result') or result.get('answer')
        sources = result.get('source_documents') or result.get('source_documents', [])
        print("\nBot:\n", answer)
        print("\n--- Retrieved sources ---")
        for i, s in enumerate(sources[:5]):
            meta = getattr(s, 'metadata', {})
            print(f"{i+1} - source: {meta.get('source')} | text excerpt: {s.page_content[:180].replace('\n',' ')}...\n")

# To run interactively, uncomment below
# chat_repl()

# %% [markdown]
## 11) Testing the chain programmatically

# You can run a test query and inspect the output programmatically (good for unit tests)

# %%

def test_query(q: str):
    out = qa_chain(q)
    print("Answer:\n", out.get('result') or out.get('answer'))
    print("\nSources:\n")
    for sd in out.get('source_documents', [])[:3]:
        print('-', sd.metadata.get('source'), '-->', sd.page_content[:140].replace('\n',' '))

# Example:
# test_query("What is LangChain and how does RAG work?")

# %% [markdown]
## 12) Improving the system

# Tips and expansions:
# - Use a context window: only pass top-n retrieved docs and summarize them when context would exceed model limits.
# - Use `chain_type='map_reduce'` or 'refine' for longer contexts / more robust responses.
# - Add conversational memory with `ConversationBufferMemory` to maintain chat history.
# - Add source citation formatting and confidence scoring.
# - For production, consider a dedicated vector DB (Milvus, Pinecone, Weaviate) or managed Postgres with proper backups.
# - Add logging, rate limiting, user authentication, and model request throttling.

# %% [markdown]
## 13) Production considerations & security

# - Do NOT expose Ollama's REST endpoint publicly without authentication. Use an API gateway, mTLS, firewalling, or bind to localhost only.
# - Secure Postgres with a strong password and network-level protections.
# - Monitor your model resource usage (RAM/VRAM). Large models require significant resources.
# - If using hosted embeddings (OpenAI), keep API keys secret and rotate them regularly.

# %% [markdown]
## 14) Troubleshooting

# Common issues:
# - "pgvector extension not found" -> ensure the DB image includes pgvector (use pgvector/pgvector images) and run CREATE EXTENSION vector;
# - "langchain-postgres import error" -> pip install langchain-postgres
# - "Ollama connection refused" -> ensure `ollama serve` is running and that models are pulled/loaded.

# %% [markdown]
## Appendix: Example SQL (for manual PGVector setup)

# Example SQL you can run in psql to create a simple table with pgvector if you want manual control:
#
# CREATE EXTENSION IF NOT EXISTS vector;
# CREATE TABLE IF NOT EXISTS documents (
#   id SERIAL PRIMARY KEY,
#   content TEXT,
#   metadata JSONB,
#   embedding VECTOR(1536) -- adjust dimension to match your embedding model
# );
# CREATE INDEX IF NOT EXISTS embedding_idx ON documents USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);

# Insert example (psycopg or sqlalchemy recommended for production)

# %% [markdown]
## Done

# This notebook is a working blueprint. Replace paths/models/collection names with your real dataset and tune chunk_size, model choices, and chain types depending on the data size and response quality needs.

# Happy building! 🎉
