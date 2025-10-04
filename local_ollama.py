import json
import ollama


def get_embedding(text: str):
    try:
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        return response["embedding"]
    except Exception as e:
        raise ConnectionError(f"Failed to get embeddings from Ollama: {e}")

def ask_ollama(context: str, question: str):
    try:
        system_prompt = """
        You are a strict RAG assistant. Use ONLY the context provided to answer the question. Follow these steps:
        1. Analyze the context and question carefully.
        2. Identify relevant information in the context that directly addresses the question.
        3. If the context does not contain enough information, state: "I cannot answer that based on the provided documents."
        4. Provide a clear and concise answer in JSON format with the following structure:
           {
             "answer": "Your answer here",
             "confidence": "A value between 0 and 1 indicating confidence in the answer",
             "source": "Relevant snippet from the context or 'None' if no relevant information"
           }
        """
        response = ollama.chat(
            model="llama3.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ]
        )
        
        return json.loads(response["message"]["content"].strip())
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response from Ollama: {e}")
        return {
            "answer": "Sorry, I received a malformed response from the language model. Please try again.",
            "confidence": "0",
            "source": "None"
        }
    except Exception as e:
        raise ConnectionError(f"Failed to get chat response from Ollama: {e}")