#!/usr/bin/env python3
"""
Database migration script to add message_type and conversation_id columns
to the conversation_messages table.
"""

import os
from sqlalchemy import text
from db import engine, SessionLocal

def migrate_database():
    """Add new columns to conversation_messages table"""
    
    print("🔄 Starting database migration...")
    
    try:
        with engine.connect() as connection:
            # Check if columns already exist
            result = connection.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'conversation_messages' 
                AND column_name IN ('message_type', 'conversation_id')
            """))
            
            existing_columns = [row[0] for row in result]
            
            # Add message_type column if it doesn't exist
            if 'message_type' not in existing_columns:
                print("📝 Adding message_type column...")
                connection.execute(text("""
                    ALTER TABLE conversation_messages 
                    ADD COLUMN message_type VARCHAR(20) DEFAULT 'user'
                """))
                print("✅ message_type column added successfully")
            else:
                print("ℹ️ message_type column already exists")
            
            # Add conversation_id column if it doesn't exist
            if 'conversation_id' not in existing_columns:
                print("📝 Adding conversation_id column...")
                connection.execute(text("""
                    ALTER TABLE conversation_messages 
                    ADD COLUMN conversation_id VARCHAR(100)
                """))
                print("✅ conversation_id column added successfully")
            else:
                print("ℹ️ conversation_id column already exists")
            
            # Update existing records to have proper message_type
            print("🔄 Updating existing records...")
            
            # Set all existing messages as user messages (since they were user questions)
            connection.execute(text("""
                UPDATE conversation_messages 
                SET message_type = 'user' 
                WHERE message_type IS NULL OR message_type = ''
            """))
            
            # Generate conversation IDs for existing records
            # Group by user_id and timestamp to create conversation pairs
            connection.execute(text("""
                UPDATE conversation_messages 
                SET conversation_id = CONCAT(user_id, '_', EXTRACT(EPOCH FROM created_at)::bigint)
                WHERE conversation_id IS NULL
            """))
            
            connection.commit()
            print("✅ Existing records updated successfully")
            
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        raise
    
    print("🎉 Database migration completed successfully!")

if __name__ == "__main__":
    migrate_database()

