import json
import asyncio
from typing import Dict, List, Any
from fastapi import WebSocket
from datetime import datetime


class ConnectionManager:
    def __init__(self):
        # Store active connections per user_id
        self.active_connections: Dict[str, List[WebSocket]] = {}
        # Store user session info
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        """Add a new WebSocket connection for a user"""
        await websocket.accept()
        
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        
        self.active_connections[user_id].append(websocket)
        
        # Track user session
        self.user_sessions[user_id] = {
            "connected_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "connection_count": len(self.active_connections[user_id])
        }
        
        print(f"User {user_id} connected. Total connections: {len(self.active_connections[user_id])}")
    
    async def disconnect(self, websocket: WebSocket, user_id: str):
        """Remove a WebSocket connection for a user"""
        if user_id in self.active_connections:
            if websocket in self.active_connections[user_id]:
                self.active_connections[user_id].remove(websocket)
            
            # Remove user if no more connections
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
                if user_id in self.user_sessions:
                    del self.user_sessions[user_id]
        
        print(f"User {user_id} disconnected. Remaining connections: {len(self.active_connections.get(user_id, []))}")
    
    async def send_personal_message(self, message: str, user_id: str):
        """Send a message to all connections of a specific user"""
        if user_id in self.active_connections:
            # Update last activity
            if user_id in self.user_sessions:
                self.user_sessions[user_id]["last_activity"] = datetime.utcnow()
            
            print(f"📤 Sending message to {len(self.active_connections[user_id])} connections for user {user_id}")
            
            # Send to all user's connections
            disconnected_websockets = []
            for i, connection in enumerate(self.active_connections[user_id]):
                try:
                    await connection.send_text(message)
                    print(f"✅ Message sent to connection {i+1} for user {user_id}")
                except Exception as e:
                    print(f"❌ Error sending message to user {user_id} connection {i+1}: {e}")
                    disconnected_websockets.append(connection)
            
            # Clean up disconnected websockets
            for ws in disconnected_websockets:
                await self.disconnect(ws, user_id)
        else:
            print(f"⚠️ No active connections found for user {user_id}")
    
    async def broadcast_to_user(self, message: Dict[str, Any], user_id: str):
        """Broadcast a JSON message to all connections of a specific user"""
        message_json = json.dumps(message)
        print(f"🔍 Broadcasting to user {user_id}: {message_json}")
        await self.send_personal_message(message_json, user_id)
    
    def get_user_connections_count(self, user_id: str) -> int:
        """Get the number of active connections for a user"""
        return len(self.active_connections.get(user_id, []))
    
    def get_all_active_users(self) -> List[str]:
        """Get list of all users with active connections"""
        return list(self.active_connections.keys())
    
    def get_total_connections(self) -> int:
        """Get total number of active connections across all users"""
        return sum(len(connections) for connections in self.active_connections.values())


# Global connection manager instance
manager = ConnectionManager()


