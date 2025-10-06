import os
import pickle
import json
from sqlalchemy import create_engine, text
from local_ollama import get_embedding, ask_ollama
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

cross_encoder_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def ask_with_context(question: str):

    docs = search_similar(question, k=20)
    
    if not docs:
        print("-> No documents found.")
        return {
            "answer": "I cannot answer that based on the provided documents.",
            "confidence": "0",
            "source": "None"
        }
    
    context = "\n\n".join(docs)
    return ask_ollama(context, question)

def search_similar(query: str, k: int = 20):
    try:
        hyde_prompt = f"Provide a concise hypothetical answer to the following question: {query}"

        hyde_response_template = ask_ollama(context="", question=hyde_prompt)
        hypothetical_answer = hyde_response_template.get("answer", "")
        
        query_emb = get_embedding(hypothetical_answer)
        
        db_url = os.environ.get("DATABASE_URL", "postgresql+psycopg2://postgres:123456@localhost:5432/mydatabase")
        engine = create_engine(db_url)
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT content, embedding <-> (:query_embedding)::vector AS distance
                    FROM pdf_embeddings
                    ORDER BY distance
                    LIMIT :k;
                """),
                {"query_embedding": query_emb, "k": k}
            )
            vector_results = [row.content for row in result]
        
        bm25_results = []
        if os.path.exists("bm25_index.pkl"):
            with open("bm25_index.pkl", "rb") as f:
                bm25_index = pickle.load(f)
            bm25_results = [doc.page_content for doc in bm25_index.get_relevant_documents(query)]
        
        combined_candidates = list(set(vector_results + bm25_results))
        
        if not combined_candidates:
            return []

        pairs = [[query, doc] for doc in combined_candidates]
        scores = cross_encoder_reranker.predict(pairs)
        
        reranked_docs = [doc for _, doc in sorted(zip(scores, combined_candidates), reverse=True)]
        
        return reranked_docs[:10]
    except Exception as e:
        print(f"Search failed: {e}")
        return []
