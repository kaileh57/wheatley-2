from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Dict, Any, Optional
from pydantic import BaseModel
import asyncio
import os
import json

from config import settings
from auth import verify_password, create_access_token, get_current_user
from models import Base
from agent_manager import AgentManager
from memory_system import AdvancedMemorySystem
from task_router import TaskRouter

# Request/Response models
class QueryRequest(BaseModel):
    query: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str

# Database setup
engine = create_engine(f"sqlite:///{settings.db_path}")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

# Global instances
agent_manager = AgentManager()
task_router = None
memory_system = None

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global task_router, memory_system
    
    # Initialize systems
    db = SessionLocal()
    memory_system = AdvancedMemorySystem(db, settings.openai_api_key)
    task_router = TaskRouter(agent_manager)
    
    # Create required directories
    os.makedirs(settings.sandbox_path, exist_ok=True)
    os.makedirs(f"{settings.sandbox_path}/global", exist_ok=True)
    os.makedirs(f"{settings.sandbox_path}/agents", exist_ok=True)
    os.makedirs(os.path.dirname(settings.db_path), exist_ok=True)
    os.makedirs(settings.memory_path, exist_ok=True)
    
    yield
    
    # Shutdown
    # Stop all agents
    for agent_id in list(agent_manager.agents.keys()):
        await agent_manager.stop_agent(agent_id)
    db.close()

# Create FastAPI app
app = FastAPI(
    title="Wheatley 2.0",
    description="Autonomous AI Assistant with MCP Integration",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes

@app.post("/token", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login endpoint"""
    if not verify_password(form_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": "user"})
    return TokenResponse(access_token=access_token, token_type="bearer")

@app.get("/")
async def root():
    """Health check"""
    return {"status": "online", "version": "2.0.0"}

@app.post("/query")
async def process_query(
    request: QueryRequest,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Process a user query"""
    query = request.query
    
    # Store in memory system
    await memory_system.learn_from_conversation([
        {"role": "user", "content": query}
    ])
    
    # Get relevant context
    context = await memory_system.get_relevant_context(query)
    
    # Route the task
    result = await task_router.route_task(query)
    
    # Store response in memory
    if result["type"] == "simple":
        await memory_system.learn_from_conversation([
            {"role": "user", "content": query},
            {"role": "assistant", "content": result["response"]}
        ])
    
    return {
        "query": query,
        "result": result,
        "context": context
    }

@app.get("/agents")
async def list_agents(current_user: str = Depends(get_current_user)):
    """List all agents"""
    agents = []
    for agent_id, agent in agent_manager.agents.items():
        status = await agent_manager.get_agent_status(agent_id)
        agents.append(status)
    return {"agents": agents}

@app.get("/agents/{agent_id}")
async def get_agent(
    agent_id: str,
    current_user: str = Depends(get_current_user)
):
    """Get specific agent status"""
    status = await agent_manager.get_agent_status(agent_id)
    if "error" in status:
        raise HTTPException(status_code=404, detail=status["error"])
    return status

@app.delete("/agents/{agent_id}")
async def stop_agent(
    agent_id: str,
    current_user: str = Depends(get_current_user)
):
    """Stop an agent"""
    await agent_manager.stop_agent(agent_id)
    return {"message": f"Agent {agent_id} stopped"}

@app.get("/memory/profile")
async def get_user_profile(
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user profile from memory system"""
    from models import UserProfile
    
    profile_entries = db.query(UserProfile).all()
    profile = {}
    for entry in profile_entries:
        try:
            profile[entry.key] = json.loads(entry.value) if entry.value.startswith('[') else entry.value
        except json.JSONDecodeError:
            profile[entry.key] = entry.value
    
    return {"profile": profile}

@app.post("/memory/search")
async def search_memories(
    request: QueryRequest,
    top_k: int = 5,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Search memories"""
    context = await memory_system.get_relevant_context(request.query, top_k)
    return context

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )