from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.vectorstores import VectorStore
from langchain.schema import Document
import faiss
from sklearn.preprocessing import normalize
import time
import numpy as np
from vars import INPUT_FILE, VECTOR_DB_PATH
import os

def create_vectorstore(user_id:str, faq_path=INPUT_FILE, persist_directory=VECTOR_DB_PATH):

    faq_path = os.path.join("user_data", user_id, "faq_data", "faqs.txt")
    persist_directory = os.path.join("user_data", user_id, "vectorstore")

    print(f"Creating vectorstore from {faq_path} to {persist_directory}")
    try:
        # Step 1: Read and parse the FAQ file
        if not os.path.exists(faq_path):
            return {"status": "error", "code": 1, "message": "FAQ file not found."}

        with open(faq_path, "r", encoding="utf-8") as f:
            content = f.read()

        blocks = content.strip().split("\n\n")
        documents = []

        for block in blocks:
            lines = block.strip().splitlines()
            question = answer = link = None
            for line in lines:
                if line.startswith("Q:"):
                    question = line[2:].strip()
                elif line.startswith("A:"):
                    answer = line[2:].strip()
                elif line.startswith("L:"):
                    link = line[2:].strip()

            if question and answer:
                doc = Document(
                    page_content=question,
                    metadata={
                        "answer": answer,
                        "link": link if link else ""
                    }
                )
                documents.append(doc)
        print(f"Parsed {len(documents)} documents from the FAQ file.")
        if not documents:
            return {"status": "error", "code": 2, "message": "No documents parsed from the FAQ file."}

        # Step 2: Get embedding model
        OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_HOST)
        print(f"Using embedding model: {embeddings}")
        # Step 3: Create FAISS vectorstore
        vectorstore = FAISS.from_documents(documents, embeddings)
        faiss_index = vectorstore.index
        print(f"Created FAISS vectorstore with {faiss_index.ntotal} vectors.")
        # Step 4: Normalize vectors
        vectors = np.array([faiss_index.reconstruct(i) for i in range(faiss_index.ntotal)]).astype("float32")
        faiss.normalize_L2(vectors)
        dimension = faiss_index.d
        new_index = faiss.IndexFlatL2(dimension)
        new_index.add(vectors)
        vectorstore.index = new_index
    
        # Step 5: Save index
        vectorstore.save_local(persist_directory)
        print(f"Saved vectorstore to {persist_directory}")
        return {"status": "success", "code": 0, "message": "Vectorstore created successfully."}

    except Exception as e:
        return {"status": "error", "code": 99, "message": f"Unexpected error: {str(e)}"}

                
