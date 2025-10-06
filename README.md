# Rag - Langchain - Vector Database

This project implements a sophisticated Retrieval-Augmented Generation (RAG) chatbot that runs locally using Ollama and a PostgreSQL database. It leverages advanced techniques like hybrid search (combining vector and keyword search), Hypothetical Document Embeddings (HyDE), and a cross-encoder for re-ranking to provide accurate, context-aware answers from your private PDF documents.

## Key Features of the Project

  * **Local First**: All components, including the Large Language Model (Ollama), embedding model, and database, can run on your local machine, ensuring data privacy.
  * **Hybrid Search**: Combines dense vector search (for semantic meaning) with sparse BM25 keyword search (for lexical matching) to retrieve a comprehensive set of candidate documents.
  * **Hypothetical Document Embeddings (HyDE)**: Improves vector search relevance by first generating a hypothetical answer to the user's query and using that answer's embedding for the search.
  * **Cross-Encoder Re-ranking**: After retrieving documents, a more powerful cross-encoder model re-ranks them for maximum relevance to the user's query, significantly boosting answer quality.
  * **Dynamic Document Upload**: Easily add new PDF documents to the knowledge base at runtime without restarting the application.
  * **PostgreSQL with `pgvector`**: Uses a robust PostgreSQL database with the `pgvector` extension for efficient and scalable vector storage and search.

## ⚙️ System Architecture

The chatbot operates through a clear, multi-stage process:

1.  **Ingestion (Seeding)**: PDF documents in the `./pdfs` directory are parsed, split into chunks, and converted into embeddings. The text content and its corresponding embedding are stored in a PostgreSQL database. A separate BM25 index is created and saved locally (`bm25_index.pkl`).
2.  **User Query**: The user asks a question via the command-line interface in `main.py`.
3.  **Retrieval (`agent.py`)**:
      * **HyDE**: The system asks the LLM to generate a hypothetical answer to the query.
      * **Vector Search**: The embedding of the hypothetical answer is used to find semantically similar document chunks from PostgreSQL.
      * **BM25 Search**: The original query is used to find keyword-relevant document chunks using the BM25 index.
      * **Combine & Re-rank**: The results from both search methods are combined, and a `CrossEncoder` model re-ranks the candidates to find the top 5 most relevant documents.
4.  **Generation (`local_ollama.py`)**: The top-ranked documents are compiled into a single context. This context, along with the original question, is passed to the Ollama LLM (`granite3.2-vision:2b`).
5.  **Structured Output**: The LLM is prompted to provide a strict JSON output containing the answer, a confidence score, and the source snippet, ensuring the answer is based solely on the provided context.
6.  **Response**: The final answer is displayed to the user.

## 🔧 Setup and Installation

### Prerequisites

  * Python 3.8+
  * PostgreSQL with the `pgvector` extension installed.
  * Ollama installed and running.
  * Ollama models pulled:
    ```bash
    ollama pull granite3.2-vision:2b
    ollama pull nomic-embed-text
    ```


### 1\. Install Dependencies

It's recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 2\. Configure the Database

In `seed.py`, update the `CONNECTION_STRING` with your PostgreSQL credentials:

```python
# seed.py
CONNECTION_STRING = "postgresql+psycopg2://<USER>:<PASSWORD>@<HOST>:<PORT>/<DATABASE_NAME>"
```

### 3\. Add Documents

Create a directory named `pdfs` in the root of the project and place your PDF files inside it.

## How to Run

1.  **Start the Chatbot**:

    ```bash
    python main.py
    ```

    On the first run, the script will automatically:

      * Connect to the database.
      * Create the necessary table (`pdf_embeddings`).
      * Process all PDFs in the `./pdfs` directory.
      * Generate and save the `bm25_index.pkl` file.
        This initial seeding may take some time depending on the number and size of your documents.

2.  **Interact with the Chatbot**:

      * **Ask a question**: Simply type your question and press Enter.
      * **Upload a new document**: Type `upload` and press Enter. You will be prompted for the filename of a PDF located in the `./pdfs` directory.
      * **Exit**: Type `exit` to close the application.

## File Breakdown

Here is a detailed explanation of each file in the project.

### `main.py`

This is the main entry point and user interface for the application.

  * **Purpose**: To handle user interaction, manage the application loop, and orchestrate calls to the seeding and agent modules.
  * **Functionality**:
      * **Initialization**: Checks if a BM25 index (`bm25_index.pkl`) exists. If not, it calls `seed_database()` from `seed.py` to process the initial set of documents.
      * **Main Loop**: Enters a continuous loop to accept user input.
      * **Command Handling**:
          * `exit`: Terminates the program.
          * `upload`: Prompts the user for a PDF filename, validates its existence in the `./pdfs` folder, and calls `seed_single_document()` to process and add it to the knowledge base.
          * **(Default) Question**: Any other input is treated as a question and is passed to `ask_with_context()` from `agent.py`.
      * **Output**: Prints the answer, confidence score, and source returned by the agent in a user-friendly format.

### `agent.py`

This file contains the core logic for the RAG agent's retrieval and answer generation process.

  * **Purpose**: To find the most relevant context for a given question and generate a reliable answer.
  * **`search_similar(query: str, k: int = 20)`**:
      * Implements the advanced hybrid search and re-ranking pipeline.
      * **HyDE**: Calls `ask_ollama` to generate a hypothetical answer to the `query`.
      * **Vector Search**: Uses the embedding of the hypothetical answer to query the PostgreSQL database for the top `k` most similar document chunks.
      * **BM25 Search**: Loads the `bm25_index.pkl` and retrieves keyword-relevant documents using the original `query`.
      * **Combine & Re-rank**: Merges the results from both searches, removes duplicates, and then uses a `CrossEncoder` model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to re-score and sort the combined candidates based on their relevance to the query.
      * Returns the top 5 most relevant document contents after re-ranking.
  * **`ask_with_context(question: str)`**:
      * Calls `search_similar()` to get the best context for the `question`.
      * If no documents are found, it returns a default "not found" message.
      * Joins the retrieved documents into a single `context` string.
      * Passes the `context` and `question` to `ask_ollama()` to generate the final, structured answer.

### `local_ollama.py`

This module is a dedicated interface for all interactions with the local Ollama instance.

  * **Purpose**: To abstract away the details of calling the Ollama API for embeddings and chat completions.
  * **`get_embedding(text: str)`**:
      * Takes a string of text.
      * Calls the Ollama embeddings API using the `nomic-embed-text` model.
      * Returns the resulting embedding vector.
      * Includes error handling for connection issues.
  * **`ask_ollama(context: str, question: str)`**:
      * Takes a `context` string and a `question` string.
      * Constructs a detailed `system_prompt` that instructs the `llama3.1` model to act as a strict RAG assistant. This prompt is crucial as it forces the model to base its answer *only* on the provided context and to respond in a specific JSON format.
      * Sends the request to the Ollama chat API.
      * Parses the JSON response from the model.
      * Includes robust error handling for JSON parsing errors or other connection issues, ensuring the application doesn't crash on a malformed LLM response.

### `seed.py`

This file handles all aspects of data ingestion, processing, embedding, and storage.

  * **Purpose**: To build and maintain the knowledge base from PDF documents.
  * **`seed_database()`**:
      * The main function for the initial, one-time setup.
      * It drops any existing `pdf_embeddings` table to ensure a clean start.
      * It creates a new `pdf_embeddings` table with columns for `id`, `content`, and a `vector(768)` to store the embeddings.
      * It iterates through all `.pdf` files in the `./pdfs` directory and calls `seed_single_document()` for each one.
      * After processing all documents, it builds the `BM25Retriever` index and saves it to `bm25_index.pkl` using `pickle`.
  * **`seed_single_document(file_path)`**:
      * Loads a single PDF file using `PyPDFLoader`.
      * Splits the document into manageable chunks using `RecursiveCharacterTextSplitter`.
      * Appends the document splits to a global list used for building the BM25 index.
      * Iterates through each chunk and calls `insert_chunk()` to add it to the database.
  * **`insert_chunk(content: str)`**:
      * Takes the text content of a single chunk.
      * Calls `get_embedding()` from `local_ollama.py` to generate its vector embedding.
      * Executes an SQL `INSERT` command to store the content and its embedding in the PostgreSQL `pdf_embeddings` table.
