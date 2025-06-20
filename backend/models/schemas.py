"""
Data models for Wheatley 2.0
"""
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """Types of WebSocket messages"""
    AUDIO = "audio"
    TEXT = "text"
    COMMAND = "command"
    RESPONSE = "response"
    STATUS = "status"
    ERROR = "error"
    TASK_UPDATE = "task_update"
    NOTIFICATION = "notification"


class TaskStatus(str, Enum):
    """Status of asynchronous tasks"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WebSocketMessage(BaseModel):
    """WebSocket message format"""
    type: MessageType
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    message_id: Optional[str] = None


class AudioMessage(BaseModel):
    """Audio data message"""
    audio_data: bytes
    sample_rate: int = 16000
    format: str = "wav"


class TextMessage(BaseModel):
    """Text message from user"""
    text: str
    context: Optional[Dict[str, Any]] = None


class CommandMessage(BaseModel):
    """Command message for system control"""
    command: str
    args: Optional[Dict[str, Any]] = None


class ResponseMessage(BaseModel):
    """AI response message"""
    text: str
    audio_url: Optional[str] = None
    tool_results: Optional[Dict[str, Any]] = None
    thinking_time: Optional[float] = None


class Memory(BaseModel):
    """Memory storage model"""
    id: Optional[int] = None
    content: str
    embedding: Optional[List[float]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    context: Optional[Dict[str, Any]] = None
    importance: float = 0.5
    category: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class Task(BaseModel):
    """Asynchronous task model"""
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    progress: float = 0.0


class UserContext(BaseModel):
    """Current user context"""
    location: str = "unknown"
    activity: str = "idle"
    last_interaction: Optional[datetime] = None
    mood: str = "neutral"
    energy_level: str = "normal"
    current_time: datetime = Field(default_factory=datetime.now)
    environment: Dict[str, Any] = Field(default_factory=dict)


class PersonalPattern(BaseModel):
    """User behavior pattern"""
    pattern_type: str  # work_hours, sleep_schedule, meal_times, etc.
    time_ranges: List[List[int]]  # List of [start_hour, end_hour]
    confidence: float = 0.5
    observations: int = 0


class ToolCall(BaseModel):
    """Tool call request/response"""
    tool_name: str
    parameters: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None


class Conversation(BaseModel):
    """Conversation history"""
    id: str
    messages: List[Dict[str, Any]]
    started_at: datetime = Field(default_factory=datetime.now)
    context: Optional[UserContext] = None
    summary: Optional[str] = None 