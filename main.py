from fastapi import FastAPI, Form, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import time
from fastapi.middleware.cors import CORSMiddleware
import faiss
import os
import numpy as np
import asyncio
import aiofiles
import json
from datetime import datetime
from vars import VECTOR_DB_PATH, OUTPUT_FILE
from db import SessionLocal, init_db, User, Faq, ConversationMessage
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from openai import OpenAI
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi import UploadFile, File, HTTPException
from dotenv import load_dotenv
from onboard import train
from websocket_manager import manager

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

@app.on_event("startup")
def _startup() -> None:
    init_db()

class Question(BaseModel):
    question: str

class OnboardUserRequest(BaseModel):
    user_id: str
    email: str | None = None
    display_name: str | None = None
    test: bool | None = None
    message: str | None = None

class TrainRequest(BaseModel):
    user_id: str
    content: str
    additionalProp1: dict = {}
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "content": "This is our company information about products and services...",
                "additionalProp1": {"metadata": "any additional data"}
            }
        }

class AskQuestionRequest(BaseModel):
    user_id: str
    comp_name: str
    specialization: str
    question: str
    sender_name:str
    sender_number:str
    time_stamp:str

class UserIdRequest(BaseModel):
    user_id: str

origins = [
    "http://localhost:3000",  # React frontend running locally
    "http://localhost:3001",  # Alternative React port
    "http://localhost:5000",  # Alternative development port
    "http://127.0.0.1:3000",  # Alternative localhost format
    "http://127.0.0.1:3001",  # Alternative localhost format
    "http://127.0.0.1:5000",  # Alternative localhost format
    "https://nimbleai.in",  # Example: Deployed frontend
    "http://nimbleai-dev.s3-website.ap-south-1.amazonaws.com",  # S3 frontend
    "https://nimbleai-dev.s3-website.ap-south-1.amazonaws.com",  # S3 frontend HTTPS
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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Server is running"}

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
    sender_name = request.sender_name
    sender_number = request.sender_number
    time_stamp = request.time_stamp

    print(f"sender_name: {sender_name}")
    print(f"sender_number: {sender_number}")
    print(f"time_stamp: {time_stamp}")

    vectorstore_path = os.path.join("vector_db", request.user_id)

    if not os.path.isdir(vectorstore_path):
        raise HTTPException(status_code=400, detail="Vectorstore not found for user. Please train first.")
    vectorstore = FAISS.load_local(
        folder_path=vectorstore_path,
        embeddings=OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_HOST),
        allow_dangerous_deserialization=True
    )

    docs = vectorstore.docstore._dict


    start_time = time.time()
    readable_time = datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")
    
    # Generate a unique conversation ID for this user question and AI response pair
    conversation_id = f"{request.user_id}_{int(start_time)}"
    
    new_message = None
    try:
        db = SessionLocal()
        new_message = ConversationMessage(
            user_id=request.user_id, 
            message=request.question,
            sender_name=request.sender_name,
            sender_number=request.sender_number,
            time_stamp=request.time_stamp,
            message_type="user",  # Mark as user message
            conversation_id=conversation_id
        )
        db.add(new_message)
        db.commit()
        db.refresh(new_message)
        
        # Broadcast the new message to Live Chat page via Lambda
        try:
            new_message_data = {
                "type": "new_message",
                "id": new_message.id,
                "message": new_message.message,
                "sender_name": new_message.sender_name,
                "sender_number": new_message.sender_number,
                "time_stamp": new_message.time_stamp,
                "created_at": new_message.created_at.isoformat() if new_message.created_at else None,
                "user_id": request.user_id,
                "message_type": "user",
                "conversation_id": conversation_id
            }
            
            # Call Lambda to broadcast to all connected clients
            import boto3
            lambda_client = boto3.client('lambda', region_name='ap-south-1')
            
            payload = {
                "action": "broadcast",
                "user_id": request.user_id,
                "message_data": new_message_data
            }
            
            lambda_client.invoke(
                FunctionName='arn:aws:lambda:ap-south-1:771397278348:function:nimbleai-websocket-handler',
                InvocationType='Event',  # Asynchronous
                Payload=json.dumps(payload)
            )
            print(f"✅ Lambda invoked for broadcasting new message to user {request.user_id}")
        except Exception as e:
            print(f"❌ Error invoking Lambda for broadcast: {e}")
            print(f"⚠️ No direct WebSocket connections available for user {request.user_id}")
            
    finally:
        db.close()
    
#    response = qa_chain.run(q.question)
    question = request.question
    print(f"OLLAMA_HOST: {OLLAMA_HOST}")
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_HOST)
    query_embedding = embeddings.embed_query(question)
    # Convert to 2D float32 array
    query_vector = np.array([query_embedding], dtype="float32")  # shape (1, d)

    # Normalize in-place
    faiss.normalize_L2(query_vector)
    k = 3  # number of nearest neighbors
    distances, indices = vectorstore.index.search(query_vector, k)
    
    # Get multiple relevant chunks instead of just one
    relevant_chunks = []
    for i in range(k):
        if indices[0][i] != -1:  # Check for valid index (-1 means no result)
            doc_id = vectorstore.index_to_docstore_id[indices[0][i]]
            document = docs[doc_id]
            relevant_chunks.append({
                'content': document.page_content,
                'metadata': document.metadata,
                'distance': distances[0][i]
            })
    
    # Combine the most relevant chunks
    combined_content = "\n\n".join([chunk['content'] for chunk in relevant_chunks])
    combined_metadata = {
        'chunk_count': len(relevant_chunks),
        'distances': [chunk['distance'] for chunk in relevant_chunks]
    }

    openai_question = question + "\n" + "\n if suitable to user's question/greeting use following info: " + combined_content
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

    # Store the AI response in the database
    try:
        db = SessionLocal()
        ai_message = ConversationMessage(
            user_id=request.user_id, 
            message=ans,
            sender_name=request.sender_name,  # Same sender_name as the question
            sender_number=request.sender_number,
            time_stamp=str(int(end_time)),
            message_type="ai",  # Mark as AI response
            conversation_id=conversation_id  # Same conversation ID as the user question
        )
        db.add(ai_message)
        db.commit()
        db.refresh(ai_message)
        
        # Broadcast the AI response to Live Chat page via Lambda
        try:
            ai_message_data = {
                "type": "new_message",
                "id": ai_message.id,
                "message": ai_message.message,
                "sender_name": ai_message.sender_name,
                "sender_number": ai_message.sender_number,
                "time_stamp": ai_message.time_stamp,
                "created_at": ai_message.created_at.isoformat() if ai_message.created_at else None,
                "user_id": request.user_id,
                "message_type": "ai",
                "conversation_id": conversation_id
            }
            
            # Call Lambda to broadcast to all connected clients
            import boto3
            lambda_client = boto3.client('lambda', region_name='ap-south-1')
            
            payload = {
                "action": "broadcast",
                "user_id": request.user_id,
                "message_data": ai_message_data
            }
            
            lambda_client.invoke(
                FunctionName='arn:aws:lambda:ap-south-1:771397278348:function:nimbleai-websocket-handler',
                InvocationType='Event',  # Asynchronous
                Payload=json.dumps(payload)
            )
            print(f"✅ Lambda invoked for broadcasting AI response to user {request.user_id}")
        except Exception as e:
            print(f"❌ Error invoking Lambda for AI response broadcast: {e}")
            print(f"⚠️ No direct WebSocket connections available for user {request.user_id}")
            
    finally:
        db.close()
    
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
    try:
        db = SessionLocal()
        rows = db.query(Faq).filter(Faq.user_id == request.user_id).order_by(Faq.id.asc()).all()
        blocks: list[str] = []
        for r in rows:
            block_lines = [
                f"Q: {r.question}",
                f"A: {r.answer}"
            ]
            if r.link:
                block_lines.append(f"L: {r.link}")
            blocks.append("\n".join(block_lines))
        return "\n\n".join(blocks)
    finally:
        db.close()


@app.post("/user_messages", response_class=PlainTextResponse)
async def get_user_messages(request: UserIdRequest):
    try:
        db = SessionLocal()
        rows = (
            db.query(ConversationMessage)
            .filter(ConversationMessage.user_id == request.user_id)
            .order_by(ConversationMessage.created_at.asc())
            .all()
        )
        messages = []
        for m in rows:
            timestamp = m.time_stamp or m.created_at.strftime("%Y-%m-%d %H:%M:%S")
            sender = m.sender_name or "Unknown"
            messages.append(f"[{timestamp}] {sender}: {m.message}")
        return "\n".join(messages)
    finally:
        db.close()

@app.post("/conversation_history")
async def get_conversation_history(request: UserIdRequest):
    try:
        db = SessionLocal()
        rows = (
            db.query(ConversationMessage)
            .filter(ConversationMessage.user_id == request.user_id)
            .order_by(ConversationMessage.created_at.asc())
            .all()
        )
        messages = []
        for m in rows:
            messages.append({
                "id": m.id,
                "message": m.message,
                "sender_name": m.sender_name,
                "sender_number": m.sender_number,
                "time_stamp": m.time_stamp,
                "created_at": m.created_at.isoformat() if m.created_at else None
            })
        return {"status": "success", "messages": messages}
    finally:
        db.close()

@app.post("/train")
async def post_train_faqs(request: TrainRequest):
    print(f"Received request body: {request}")
    
    # Extract user_id and content from the request
    user_id = request.user_id
    content = request.content
    
    # Handle additional properties - they can contain any extra data
    additional_props = {"additionalProp1": request.additionalProp1} if request.additionalProp1 else {}
    
    print(f"Training the model for user {user_id} with paragraph content")
    if additional_props:
        print(f"Additional properties received: {list(additional_props.keys())}")

    try:
        # Train the model directly with the content
        # No database storage needed
        train(user_id, content)
        
        return {
            "status": "success", 
            "message": "Model trained successfully",
            "additional_properties_processed": len(additional_props)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/onboard_user")
async def post_onboard_user(request: OnboardUserRequest):
    try:
        user_id = request.user_id
        email = request.email
        display_name = request.display_name

        print(f"Onboarding user: {user_id} ({email})")

        db = SessionLocal()
        try:
            existing = db.get(User, user_id)
            if existing is None:
                db.add(User(user_id=user_id, email=email, display_name=display_name))
            else:
                existing.email = email
                existing.display_name = display_name
            db.commit()
        finally:
            db.close()

        os.makedirs(os.path.join("vector_db", user_id), exist_ok=True)

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



@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time chat"""
    await manager.connect(websocket, user_id)
    
    try:
        # Send connection confirmation
        await manager.broadcast_to_user({
            "type": "connection_status",
            "status": "connected",
            "user_id": user_id,
            "message": "WebSocket connection established"
        }, user_id)
        
        # Handle incoming messages
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message_data = json.loads(data)
                message_type = message_data.get("type")
                
                if message_type == "fetch_messages":
                    # Fetch messages from database for the user
                    try:
                        db = SessionLocal()
                        rows = (
                            db.query(ConversationMessage)
                            .filter(ConversationMessage.user_id == user_id)
                            .order_by(ConversationMessage.created_at.asc())
                            .all()
                        )
                        
                        # Send database messages
                        database_messages = {
                            "type": "database_messages",
                            "messages": [
                                {
                                    "id": m.id,
                                    "message": m.message,
                                    "sender_name": m.sender_name,
                                    "sender_number": m.sender_number,
                                    "time_stamp": m.time_stamp,
                                    "created_at": m.created_at.isoformat() if m.created_at else None,
                                    "message_type": m.message_type,
                                    "conversation_id": m.conversation_id
                                }
                                for m in rows
                            ]
                        }
                        await manager.broadcast_to_user(database_messages, user_id)
                        
                    except Exception as e:
                        print(f"Error loading conversation history for user {user_id}: {e}")
                        await manager.broadcast_to_user({
                            "type": "error",
                            "message": f"Error loading messages: {str(e)}"
                        }, user_id)
                    finally:
                        db.close()
                
                elif message_type == "store_message":
                    # Store new message in database
                    message = message_data.get("message", "")
                    sender_name = message_data.get("sender_name", "User")
                    sender_number = message_data.get("sender_number", "")
                    time_stamp = message_data.get("time_stamp", str(int(time.time())))
                    
                    try:
                        db = SessionLocal()
                        # Generate conversation ID for this message
                        conversation_id = f"{user_id}_{int(time.time())}"
                        
                        new_message = ConversationMessage(
                            user_id=user_id,
                            message=message,
                            sender_name=sender_name,
                            sender_number=sender_number,
                            time_stamp=time_stamp,
                            message_type="user",
                            conversation_id=conversation_id
                        )
                        db.add(new_message)
                        db.commit()
                        db.refresh(new_message)
                        
                        # Broadcast the new message back to confirm storage
                        new_message_data = {
                            "type": "new_message",
                            "id": new_message.id,
                            "message": new_message.message,
                            "sender_name": new_message.sender_name,
                            "sender_number": new_message.sender_number,
                            "time_stamp": new_message.time_stamp,
                            "created_at": new_message.created_at.isoformat() if new_message.created_at else None,
                            "user_id": user_id,
                            "message_type": "user",
                            "conversation_id": conversation_id
                        }
                        await manager.broadcast_to_user(new_message_data, user_id)
                        
                    except Exception as e:
                        print(f"Error storing message in database: {e}")
                        await manager.broadcast_to_user({
                            "type": "error",
                            "message": f"Error storing message: {str(e)}"
                        }, user_id)
                    finally:
                        db.close()
                
                elif message_type == "question":
                    # Legacy support for question type
                    question = message_data.get("question", "")
                    comp_name = message_data.get("comp_name", "NimbleAI")
                    specialization = message_data.get("specialization", "AI chatbots")
                    sender_name = message_data.get("sender_name", "User")
                    sender_number = message_data.get("sender_number", "")
                    time_stamp = message_data.get("time_stamp", str(int(time.time())))
                    
                    # Store message in database
                    try:
                        db = SessionLocal()
                        # Generate conversation ID for this message
                        conversation_id = f"{user_id}_{int(time.time())}"
                        
                        new_message = ConversationMessage(
                            user_id=user_id,
                            message=question,
                            sender_name=sender_name,
                            sender_number=sender_number,
                            time_stamp=time_stamp,
                            message_type="user",
                            conversation_id=conversation_id
                        )
                        db.add(new_message)
                        db.commit()
                        db.refresh(new_message)
                        
                        # Broadcast the new message
                        new_message_data = {
                            "type": "new_message",
                            "id": new_message.id,
                            "message": new_message.message,
                            "sender_name": new_message.sender_name,
                            "sender_number": new_message.sender_number,
                            "time_stamp": new_message.time_stamp,
                            "created_at": new_message.created_at.isoformat() if new_message.created_at else None,
                            "user_id": user_id,
                            "message_type": "user",
                            "conversation_id": conversation_id
                        }
                        await manager.broadcast_to_user(new_message_data, user_id)
                        
                    except Exception as e:
                        print(f"Error storing message in database: {e}")
                    finally:
                        db.close()
                
                elif message_type == "ping":
                    # Handle ping messages
                    await manager.broadcast_to_user({
                        "type": "pong",
                        "timestamp": message_data.get("timestamp", time.time())
                    }, user_id)
                
                else:
                    # Unknown message type
                    await manager.broadcast_to_user({
                        "type": "error",
                        "message": f"Unknown message type: {message_type}"
                    }, user_id)
                
            except json.JSONDecodeError:
                await manager.broadcast_to_user({
                    "type": "error",
                    "message": "Invalid JSON format"
                }, user_id)
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"Error processing WebSocket message: {e}")
                await manager.broadcast_to_user({
                    "type": "error",
                    "message": f"Server error: {str(e)}"
                }, user_id)
                break
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error for user {user_id}: {e}")
    finally:
        await manager.disconnect(websocket, user_id)

@app.post("/api/websocket-proxy")
async def websocket_proxy(request: dict):
    """Proxy endpoint for Lambda function to communicate with EC2 backend"""
    try:
        user_id = request.get("user_id", "anonymous")
        message_type = request.get("type")
        
        print(f"🔗 Lambda proxy request: {message_type} for user {user_id}")
        
        if message_type == "store_message":
            # Store message in database
            message = request.get("message", "")
            sender_name = request.get("sender_name", "User")
            sender_number = request.get("sender_number", "")
            time_stamp = request.get("time_stamp", str(int(time.time())))
            
            db = SessionLocal()
            try:
                # Generate conversation ID for this message
                conversation_id = f"{user_id}_{int(time.time())}"
                
                new_message = ConversationMessage(
                    user_id=user_id,
                    message=message,
                    sender_name=sender_name,
                    sender_number=sender_number,
                    time_stamp=time_stamp,
                    message_type="user",
                    conversation_id=conversation_id
                )
                db.add(new_message)
                db.commit()
                db.refresh(new_message)
                
                # Broadcast to connected WebSocket clients via Lambda
                new_message_data = {
                    "type": "new_message",
                    "id": new_message.id,
                    "message": new_message.message,
                    "sender_name": new_message.sender_name,
                    "sender_number": new_message.sender_number,
                    "time_stamp": new_message.time_stamp,
                    "created_at": new_message.created_at.isoformat() if new_message.created_at else None,
                    "user_id": user_id,
                    "message_type": "user",
                    "conversation_id": conversation_id
                }
                
                # Call Lambda to broadcast to all connected clients
                try:
                    import boto3
                    lambda_client = boto3.client('lambda', region_name='ap-south-1')
                    
                    payload = {
                        "action": "broadcast",
                        "user_id": user_id,
                        "message_data": new_message_data
                    }
                    
                    lambda_client.invoke(
                        FunctionName='arn:aws:lambda:ap-south-1:771397278348:function:nimbleai-websocket-handler',
                        InvocationType='Event',  # Asynchronous
                        Payload=json.dumps(payload)
                    )
                    print(f"✅ Lambda invoked for broadcasting new message to user {user_id}")
                except Exception as e:
                    print(f"❌ Error invoking Lambda for broadcast: {e}")
                    print(f"⚠️ Lambda broadcasting failed for user {user_id}")
                
                return {
                    "success": True,
                    "message_id": new_message.id,
                    "message": "Message stored successfully"
                }
            finally:
                db.close()
                
        elif message_type == "fetch_messages":
            # Fetch messages from database
            db = SessionLocal()
            try:
                rows = (
                    db.query(ConversationMessage)
                    .filter(ConversationMessage.user_id == user_id)
                    .order_by(ConversationMessage.created_at.asc())
                    .all()
                )
                
                messages = [
                    {
                        "id": m.id,
                        "message": m.message,
                        "sender_name": m.sender_name,
                        "sender_number": m.sender_number,
                        "time_stamp": m.time_stamp,
                        "created_at": m.created_at.isoformat() if m.created_at else None,
                        "message_type": m.message_type,
                        "conversation_id": m.conversation_id
                    }
                    for m in rows
                ]
                
                return {
                    "success": True,
                    "messages": messages
                }
            finally:
                db.close()
        else:
            return {
                "success": False,
                "error": f"Unknown message type: {message_type}"
            }
            
    except Exception as e:
        print(f"❌ Error in websocket proxy: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/api/broadcast-message")
async def broadcast_message(request: dict):
    """Broadcast a new message to all connected WebSocket clients for a user"""
    try:
        user_id = request.get("user_id")
        message_data = request.get("message_data", {})
        
        if not user_id:
            return {
                "success": False,
                "error": "user_id is required"
            }
        
        print(f"📡 Broadcasting message to user: {user_id}")
        
        # Call Lambda function to broadcast the message
        import boto3
        lambda_client = boto3.client('lambda', region_name='ap-south-1')
        
        payload = {
            "action": "broadcast",
            "user_id": user_id,
            "message_data": message_data
        }
        
        response = lambda_client.invoke(
            FunctionName='nimbleai-websocket-handler',
            InvocationType='Event',  # Asynchronous
            Payload=json.dumps(payload)
        )
        
        print(f"✅ Lambda invoked for broadcasting: {response}")
        
        return {
            "success": True,
            "message": "Broadcast request sent to Lambda"
        }
        
    except Exception as e:
        print(f"❌ Error broadcasting message: {e}")
        return {
            "success": False,
            "error": str(e)
        }