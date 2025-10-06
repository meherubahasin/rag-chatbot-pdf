import json
import requests
import re

def get_embedding(text: str):
    try:
        # Use requests for embeddings as well for consistency and timeouts
        payload = {"model": "nomic-embed-text", "prompt": text}
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json=payload,
            timeout=60 # Reduced timeout
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except Exception as e:
        # Raise a specific ConnectionError for upstream handling
        raise ConnectionError(f"Failed to get embeddings from Ollama: {e}")

def ask_ollama(context, question, model="granite3.2-vision:2b"):
    """
    Sends a question and context to the local Ollama API and returns a robust, clean JSON response.
    """
    system_prompt = f"""
You are cooperative and accurate assistant.

You must ALWAYS return responses in valid JSON format, and never include any text outside the JSON.

Use ONLY the information from the provided context to answer the question.
If the answer is not clearly available, respond with:
"I couldn’t find that specific information in the document" and, then provide as much relevant information as possible from the context.
If the answer is not clearly available in the context, you may use your pre-existing knowledge to answer the question, but you must clearly state that you are doing so.


Output format (MANDATORY):
{{
  "answer": "string",
  "confidence": float,
  "source": "string"
}}

Example:
{{
  "answer": "There are four disciplinary actions listed: verbal warning, written warning, suspension, and termination.",
  "confidence": 1.0,
  "source": "Section 8: Disciplinary Actions"
}}

---
Context:
{context}

User question:
{question}
---
"""

    payload = {
        "model": model,
        "prompt": system_prompt
    }


    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=60, # Reduced timeout
            stream=True
        )
        response.raise_for_status()

        output_text = ""
        for line in response.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    output_text += data["response"]
            except json.JSONDecodeError:
                continue

        # --- JSON Cleanup Section ---
        cleaned = re.sub(r"[\x00-\x1F\x7F]", "", output_text)
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            cleaned = match.group(0)
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
            for key in ["answer", "confidence", "source"]:
                if key not in parsed:
                    parsed[key] = "" if key != "confidence" else 0.0
            return parsed
        except json.JSONDecodeError:
            print("⚠️ Could not parse cleaned JSON, returning raw text.")
            return {
                "answer": cleaned,
                "confidence": 0.0,
                "source": "Parsing fallback"
            }

    except Exception as e:
        print(f"⚠️ Ollama request failed: {e}")
        return {
            "answer": f"Ollama request failed: {e}",
            "confidence": 0.0,
            "source": "Network error"
        }
