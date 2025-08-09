from fastapi import FastAPI, Form
from pydantic import BaseModel
import time
from fastapi.middleware.cors import CORSMiddleware
import faiss
import os
import numpy as np
import asyncio
import aiofiles
from datetime import datetime
from vars import VECTOR_DB_PATH, OUTPUT_FILE
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from openai import OpenAI
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi import UploadFile, File, HTTPException
from dotenv import load_dotenv
from onboard import train

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable not set. Please provide it when starting the server.")

client = OpenAI(api_key=OPENAI_API_KEY)

async def save_text_async(filename: str, content: str):
    async with aiofiles.open(filename, mode='a') as f:
        await f.write(content)

def save_text_background(filename, content):
    asyncio.create_task(save_text_async(filename, content))


#import os
app = FastAPI()

class Question(BaseModel):
    question: str

class OnboardUserRequest(BaseModel):
    user_id: str
    email: str | None = None
    display_name: str | None = None
    test: bool | None = None
    message: str | None = None

class AskQuestionRequest(BaseModel):
    user_id: str
    comp_name: str
    specialization: str
    question: str

class UserIdRequest(BaseModel):
    user_id: str

origins = [
    "http://localhost:3000",  # Example: React frontend running locally
    "https://nimbleai.in",  # Example: Deployed frontend
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,  # Allow cookies or authentication headers
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")



print(f"Loaded existing FAISS index from {VECTOR_DB_PATH}")

#docs = vectorstore.docstore._dict
#index = vectorstore.index
#num_vectors = index.ntotal
#print(f"faiss  dimensions: {index.d}, Number of vectors: {num_vectors}")
#print(f"query_vector  dimensions: {query_vector.shape}")
#all_vectors = index.reconstruct_n(0, num_vectors)  # Shape: [num_vectors, dims]

@app.post("/ask")
async def ask_question(request: AskQuestionRequest):
    BASE_PROMPT = f"""
Role: You are {request.comp_name}'s chat assistant, {request.comp_name} is a company that specializes in {request.specialization}. Your task is to respond in a way that helps users understand the value of using AI chatbots.

Instructions:
1. Only use greetings when you are greeted by the user.
2. Use a friendly and approachable tone.
3. Provide clear and helpful answers to the user's question.
4. If the question is unrelated to {request.comp_name}, politely let the user know that you can only assist with {request.comp_name}-related queries.
5. Avoid using any special characters in your response.
6. Keep your response concise and focused.
7. The response must be as short as possible.
8. The answer must be according to the user's question.

User Prompt:
"""
    vectorstore_path = os.path.join("user_data", request.user_id, "vectorstore")

    user_messages_path = os.path.join("user_data", request.user_id, "conversations", "user_messages.txt")

    vectorstore = FAISS.load_local(
    folder_path = vectorstore_path,
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_HOST),
    allow_dangerous_deserialization=True  # Required for FAISS
    )

    docs = vectorstore.docstore._dict


    start_time = time.time()
    readable_time = datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")
    save_text_background(user_messages_path, readable_time + " " + request.question + '\n')
    
#    response = qa_chain.run(q.question)
    question = request.question
    print(f"OLLAMA_HOST: {OLLAMA_HOST}")
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_HOST)
    query_embedding = embeddings.embed_query(question)
    # Convert to 2D float32 array
    query_vector = np.array([query_embedding], dtype="float32")  # shape (1, d)

    # Normalize in-place
    faiss.normalize_L2(query_vector)
    k = 1  # number of nearest neighbors
    distances, indices = vectorstore.index.search(query_vector, k)
    doc_id = vectorstore.index_to_docstore_id[indices[0][0]]
    # Step 2: Fetch the document from the docstore
    document = docs[doc_id]

    openai_question = question + "\n" + "\n if suitable to user's question/greeting use following info: "+ str(document.metadata)
    oai_time_start = time.time()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content":BASE_PROMPT},
            {"role": "user", "content": openai_question}
        ]
    )
    oai_time_end = time.time()
    print(f"OpenAI response time: {oai_time_end - oai_time_start} seconds")
    print(f" openai response = {response}")

    end_time = time.time()
    ans = response.choices[0].message.content
    return {
    "answer": {
        "metadata": {
            "answer": ans,
            "processing_time": end_time - start_time
                    },
               }
            }

@app.post("/faqs", response_class=PlainTextResponse)
async def get_faqs(request: UserIdRequest):
    faqs_path = os.path.join("user_data", request.user_id, "faq_data", "faqs.txt")
    if not os.path.exists(faqs_path):
        return ""
    async with aiofiles.open(faqs_path, mode='r') as f:
        content = await f.read()
    return content

@app.post("/faqs")
async def post_faqs(request: UserIdRequest):
    # This endpoint is for JSON requests (getting FAQs)
    faqs_path = os.path.join("user_data", request.user_id, "faq_data", "faqs.txt")
    if not os.path.exists(faqs_path):
        return {"status": "success", "faqs": ""}
    async with aiofiles.open(faqs_path, mode='r') as f:
        content = await f.read()
    return {"status": "success", "faqs": content}

@app.post("/faqs/upload")
async def upload_faqs(file: UploadFile = File(...), user_id: str = Form(...)):
    # Dedicated endpoint for file uploads with user_id from FormData
    faqs_path = os.path.join("user_data", user_id, "faq_data", "faqs.txt")
    content = await file.read()
    async with aiofiles.open(faqs_path, mode='wb') as f:
        await f.write(content)
    return {"status": "success", "message": "File uploaded successfully"}

@app.post("/user_messages", response_class=PlainTextResponse)
async def get_user_messages(request: UserIdRequest):
    user_messages_path = os.path.join("user_data", request.user_id, "conversations", "user_messages.txt")
    if not os.path.exists(user_messages_path):
        return ""
    async with aiofiles.open(user_messages_path, mode='r') as f:
        content = await f.read()
    return content

@app.post("/train")
async def post_train_faqs(request: UserIdRequest):
    print(f"Training the model for user {request.user_id}")
    train(request.user_id)
    return {"status": "success"}

@app.post("/onboard_user")
async def post_onboard_user(request: OnboardUserRequest):
    try:
        user_id = request.user_id
        email = request.email
        display_name = request.display_name
        
        print(f"Onboarding user: {user_id} ({email})")
        
        user_path = os.path.join("user_data", user_id)
        os.makedirs(user_path, exist_ok=True)

        user_faqs_path = os.path.join(user_path, "faq_data")
        os.makedirs(user_faqs_path, exist_ok=True)

        user_vectorstore_path = os.path.join(user_path, "vectorstore")
        os.makedirs(user_vectorstore_path, exist_ok=True)

        user_messages_path = os.path.join(user_path, "conversations")
        os.makedirs(user_messages_path, exist_ok=True)

        return {
            "success": True,
            "message": "User onboarded successfully",
            "user_id": user_id
        }
    except Exception as e:
        print(f"Onboarding error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/test")
async def test_endpoint(request: dict):
    return {
        "success": True,
        "message": "Test endpoint working!",
        "received_data": request
    }