import os
from seed import seed_database, seed_single_document
from agent import ask_with_context
import time
from pathlib import Path

def main():
    print("Welcome to the RAG Chatbot.")
    
    if not os.path.exists("bm25_index.pkl"):
        print("-> BM25 index not found. Seeding the database...")
        seed_database()
        print("-> Database seeded successfully.")
    else:
        print("-> BM25 index found. Skipping initial seeding.")

    while True:
        user_input = input("\nAsk a question about your documents (or 'exit' to quit, 'upload' to add a new PDF): ")
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        if user_input.lower() == 'upload':
            file_name = input("Enter the filename of the PDF to upload (e.g., my_new_file.pdf): ")
            file_path = Path("./pdfs") / file_name
            
            if not file_path.is_file():
                print(f"Error: File '{file_path}' not found. Please ensure the file is in the 'pdfs' directory.")
                continue

            start_time = time.time()
            seed_single_document(file_path)
            end_time = time.time()
            print(f"Document '{file_name}' processed and added to the knowledge base in {end_time - start_time:.2f} seconds.")
            continue

        answer = ask_with_context(user_input)
        
        # **Handle the dictionary response from ask_with_context**
        print(f"\nBot: {answer.get('answer', 'No answer received.')}")
        print(f"Confidence: {answer.get('confidence', 'N/A')}")
        print(f"Source: {answer.get('source', 'N/A')}")

if __name__ == "__main__":
    main()
