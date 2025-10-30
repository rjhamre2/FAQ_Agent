import os
from datetime import datetime
from typing import Generator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from sqlalchemy import String, Text, DateTime, ForeignKey
from sqlalchemy.engine import URL
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from sqlalchemy import create_engine
import time


class Base(DeclarativeBase):
    pass


def _build_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return database_url
    # Fallback for local/dev - use your actual username
    return "postgresql+psycopg://rupeshjhamre@localhost:5432/whatsapp_chats"


DATABASE_URL = _build_database_url()

# echo can be toggled on for debugging by setting SQLALCHEMY_ECHO=true
SQLALCHEMY_ECHO = os.getenv("SQLALCHEMY_ECHO", "false").lower() == "true"

engine = create_engine(DATABASE_URL, echo=SQLALCHEMY_ECHO, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    email: Mapped[str | None] = mapped_column(String(256), nullable=True)
    display_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    faqs: Mapped[list["Faq"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    messages: Mapped[list["ConversationMessage"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Faq(Base):
    __tablename__ = "faqs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id", ondelete="CASCADE"), index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    link: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    user: Mapped[User] = relationship(back_populates="faqs")


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id", ondelete="CASCADE"), index=True)
    message: Mapped[str] = mapped_column(Text)
    sender_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    sender_number: Mapped[str | None] = mapped_column(String(50), nullable=True)
    time_stamp: Mapped[str | None] = mapped_column(String(100), nullable=True)
    message_type: Mapped[str] = mapped_column(String(20), default="user")  # "user" or "ai"
    conversation_id: Mapped[str | None] = mapped_column(String(100), nullable=True)  # To group related messages
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, index=True)

    user: Mapped[User] = relationship(back_populates="messages")


def init_db(retries: int = 30, delay_seconds: float = 1.0) -> None:
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            Base.metadata.create_all(bind=engine)
            return
        except Exception as exc:  # pragma: no cover - simple startup retry
            last_error = exc
            time.sleep(delay_seconds)
    if last_error:
        raise last_error


def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


