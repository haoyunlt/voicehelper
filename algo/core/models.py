from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class Message(BaseModel):
    role: MessageRole
    content: str

class QueryRequest(BaseModel):
    messages: List[Message]
    top_k: Optional[int] = Field(default=5, ge=1, le=20)
    temperature: Optional[float] = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=1024, ge=1, le=4096)
    filters: Optional[Dict[str, Any]] = None

class Reference(BaseModel):
    chunk_id: str
    source: str
    score: float
    content: Optional[str] = None

class QueryResponse(BaseModel):
    type: str  # "delta", "refs", "end"
    content: Optional[str] = None
    refs: Optional[List[Reference]] = None

class IngestRequest(BaseModel):
    dataset_id: str = "default"
    files: Optional[List[str]] = None
    urls: Optional[List[str]] = None
    chunk_size: Optional[int] = Field(default=600, ge=100, le=2000)
    chunk_overlap: Optional[int] = Field(default=80, ge=0, le=500)
    metadata: Optional[Dict[str, Any]] = None

class IngestResponse(BaseModel):
    task_id: str

class TaskStatus(BaseModel):
    task_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: int = 0
    message: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class VoiceQueryRequest(BaseModel):
    conversation_id: str
    audio_chunk: str  # base64 encoded audio
    seq: int
    codec: str = "opus"
    sample_rate: int = 16000

class VoiceQueryResponse(BaseModel):
    type: str  # "asr_partial", "asr_final", "llm_delta", "tts_chunk", "refs", "done", "error"
    seq: Optional[int] = None
    text: Optional[str] = None
    pcm: Optional[str] = None  # base64 encoded PCM audio
    refs: Optional[List[Reference]] = None
    error: Optional[str] = None
