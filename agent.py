import os
import pickle
import json
from sqlalchemy import create_engine, text
from local_ollama import get_embedding, ask_ollama
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

# Initialize the cross-encoder model for re-ranking
cross_encoder_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def ask_with_context(question: str):
    docs = search_similar(question, k=20)
    
    if not docs:
        return {
            "answer": "Not in the provided documents.",
            "confidence": "0",
            "source": "None"
        }
    
    context = "\n\n".join(docs)
    return ask_ollama(context, question)

def search_similar(query: str, k: int = 20):

    try:
        # Generate hypothetical answer for HyDE
        hyde_prompt = f"Provide a concise hypothetical answer to the following question: {query}"
        hypothetical_answer_response = ask_ollama(context="", question=hyde_prompt)
        hypothetical_answer = hypothetical_answer_response.get("answer", "")
        
        # Use hypothetical answer for embedding
        query_emb = get_embedding(hypothetical_answer)
        
        # Connect to the database and perform vector search
        engine = create_engine("postgresql+psycopg2://postgres:123456@localhost:5432/mydatabase")
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
        
        # Perform BM25 search (using original query)
        bm25_results = []
        if os.path.exists("bm25_index.pkl"):
            with open("bm25_index.pkl", "rb") as f:
                bm25_index = pickle.load(f)
            bm25_results = [doc.page_content for doc in bm25_index.get_relevant_documents(query)]
        
        # Combine results from both search methods
        combined_candidates = list(set(vector_results + bm25_results))
        
        if not combined_candidates:
            return []

        # Re-rank combined candidates to get the most relevant ones
        pairs = [[query, doc] for doc in combined_candidates]
        scores = cross_encoder_reranker.predict(pairs)
        
        # Sort documents by re-ranking score and return the top 5
        reranked_docs = [doc for _, doc in sorted(zip(scores, combined_candidates), reverse=True)]
        
        return reranked_docs[:5]
    except Exception as e:
        print(f"Search failed: {e}")
        return []