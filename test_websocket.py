import asyncio
import websockets
import json
import time

async def test_websocket():
    """Test WebSocket connection and messaging"""
    uri = "ws://localhost:8000/ws/test_user_001"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket!")
            
            # Listen for messages
            async def receive_messages():
                try:
                    while True:
                        message = await websocket.recv()
                        data = json.loads(message)
                        print(f"📨 Received: {json.dumps(data, indent=2)}")
                except websockets.exceptions.ConnectionClosed:
                    print("❌ WebSocket connection closed")
            
            # Start listening for messages
            receive_task = asyncio.create_task(receive_messages())
            
            # Wait a bit for connection confirmation
            await asyncio.sleep(2)
            
            # Send a test question
            test_message = {
                "type": "question",
                "question": "What is your company name?",
                "comp_name": "NimbleAI",
                "specialization": "AI chatbots",
                "sender_name": "Test User",
                "sender_number": "1234567890",
                "time_stamp": "2025-01-01 16:00:00"
            }
            
            print(f"📤 Sending: {json.dumps(test_message, indent=2)}")
            await websocket.send(json.dumps(test_message))
            
            # Wait for response
            await asyncio.sleep(5)
            
            # Cancel the receive task
            receive_task.cancel()
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 Starting WebSocket test...")
    asyncio.run(test_websocket())


