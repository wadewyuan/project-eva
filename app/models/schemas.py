from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ---------- Chat ----------

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(..., min_length=1)
    stream: bool = False


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    tone_detected: str = "default"


# ---------- Session ----------

class SessionInfo(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int


class MessageRecord(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime


# ---------- Memory ----------

class MemoryItem(BaseModel):
    id: int
    fact_text: str
    category: str
    importance_score: int
    created_at: datetime


# ---------- Persona ----------

class PersonaInfo(BaseModel):
    id: str
    name: str
    description: str
    is_active: bool = False


class PersonaDetail(BaseModel):
    id: str
    name: str
    role: str
    description: str
    tone: dict[str, str]
    speaking_style: list[str]
    personality_traits: list[str]
    forbidden: list[str]


# ---------- Voice ----------

class VoiceTranscribeResponse(BaseModel):
    text: str


class VoiceSynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1)
