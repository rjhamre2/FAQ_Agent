from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.vectorstores import VectorStore
from langchain.schema import Document
import faiss
from sklearn.preprocessing import normalize
import time
import numpy as np
from vars import INPUT_FILE, VECTOR_DB_PATH
from db import SessionLocal, Faq
import os

def create_vectorstore(user_id:str, faq_path=INPUT_FILE, persist_directory=VECTOR_DB_PATH):
    persist_directory = os.path.join(VECTOR_DB_PATH, user_id)
    os.makedirs(persist_directory, exist_ok=True)
    print(f"Creating vectorstore for user {user_id} to {persist_directory}")
    try:
        # Step 1: Read FAQs from database
        db = SessionLocal()
        try:
            rows = db.query(Faq).filter(Faq.user_id == user_id).all()
        finally:
            db.close()

        documents = []
        for row in rows:
            if row.question and row.answer:
                doc = Document(
                    page_content=row.question,
                    metadata={
                        "answer": row.answer,
                        "link": row.link or ""
                    }
                )
                documents.append(doc)
        print(f"Parsed {len(documents)} documents from the database.")
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

                
