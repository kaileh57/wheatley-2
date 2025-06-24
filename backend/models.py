from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Task(Base):
    __tablename__ = "tasks"
    
    id = Column(Integer, primary_key=True)
    query = Column(Text, nullable=False)
    status = Column(String(50), default="pending")
    result = Column(Text)
    agent_id = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    metadata = Column(JSON)

class Memory(Base):
    __tablename__ = "memories"
    
    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    embedding = Column(Text)  # Store as JSON string
    category = Column(String(100))
    importance = Column(Integer, default=5)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)

class UserProfile(Base):
    __tablename__ = "user_profile"
    
    id = Column(Integer, primary_key=True)
    key = Column(String(100), unique=True)
    value = Column(Text)
    updated_at = Column(DateTime, default=datetime.utcnow)