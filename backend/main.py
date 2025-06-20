"""
Wheatley 2.0 - Personal AI Assistant
Main FastAPI application
"""
import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from backend.config import settings
from backend.models.schemas import (
    WebSocketMessage, MessageType, TaskStatus,
    TextMessage, AudioMessage, ResponseMessage
)
from backend.services.agent import PersonalAgent
from backend.services.memory import MemoryService
from backend.services.context import ContextManager
from backend.services.tasks import TaskExecutor
from backend.services.voice import VoiceService
from backend.services.mcp_tools import MCPToolManager
from backend.hardware.device import HardwareDevice
from backend.hardware.wake_word import create_wake_word_detector
from backend.utils.auth import get_api_key, validate_websocket_key

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Wheatley 2.0",
    description="Personal AI Assistant API",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instances
memory_service: Optional[MemoryService] = None
context_manager: Optional[ContextManager] = None
tool_manager: Optional[MCPToolManager] = None
personal_agent: Optional[PersonalAgent] = None
task_executor: Optional[TaskExecutor] = None
voice_service: Optional[VoiceService] = None
hardware_device: Optional[HardwareDevice] = None
wake_word_detector = None

# WebSocket connections
active_connections: Dict[str, WebSocket] = {}


class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept new connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")
    
    def disconnect(self, client_id: str):
        """Remove connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected")
    
    async def send_message(self, client_id: str, message: WebSocketMessage):
        """Send message to specific client"""
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message.dict())
    
    async def broadcast(self, message: WebSocketMessage):
        """Broadcast message to all connected clients"""
        for client_id, connection in self.active_connections.items():
            try:
                await connection.send_json(message.dict())
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")


# Connection manager instance
connection_manager = ConnectionManager()


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global memory_service, context_manager, tool_manager
    global personal_agent, task_executor, voice_service
    global hardware_device, wake_word_detector
    
    logger.info("Starting Wheatley 2.0...")
    
    # Initialize services
    memory_service = MemoryService()
    context_manager = ContextManager()
    tool_manager = MCPToolManager()
    
    personal_agent = PersonalAgent(memory_service, tool_manager)
    task_executor = TaskExecutor(personal_agent)
    voice_service = VoiceService()
    
    # Initialize hardware (if available)
    hardware_device = HardwareDevice()
    await hardware_device.initialize()
    
    # Register task notification callback
    async def task_notification_callback(notification_type: str, data: Dict[str, Any]):
        """Send task notifications via WebSocket"""
        message = WebSocketMessage(
            type=MessageType.TASK_UPDATE if notification_type == "task_update" else MessageType.NOTIFICATION,
            data=data
        )
        await connection_manager.broadcast(message)
        
        # Play sound on hardware device for task completion
        if notification_type == "task_complete" and data.get("play_sound"):
            # This would trigger a sound on the hardware device
            logger.info("Task completed - playing notification sound")
    
    task_executor.register_notification_callback(task_notification_callback)
    
    # Initialize wake word detector
    async def on_wake_word_detected():
        """Handle wake word detection"""
        logger.info("Wake word detected!")
        
        # Set hardware to listening state
        await hardware_device.set_listening(True)
        
        # Record audio
        audio_data = await voice_service.continuous_recording_with_vad()
        
        # Process the audio
        await process_voice_input(audio_data)
        
        # Reset hardware state
        await hardware_device.set_listening(False)
    
    wake_word_detector = create_wake_word_detector(on_wake_word_detected)
    
    # Start wake word listening in background
    asyncio.create_task(wake_word_detector.start_listening())
    
    # Start proactive monitoring
    asyncio.create_task(proactive_monitoring())
    
    # Ensure required directories exist
    Path("data").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    logger.info("Wheatley 2.0 started successfully!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Wheatley 2.0...")
    
    if wake_word_detector:
        wake_word_detector.cleanup()
    
    if hardware_device:
        hardware_device.cleanup()
    
    logger.info("Wheatley 2.0 shutdown complete")


async def process_voice_input(audio_data):
    """Process voice input from hardware"""
    try:
        # Set processing state
        await hardware_device.set_processing(True)
        
        # Transcribe audio
        text, confidence = await voice_service.process_voice_command(audio_data)
        
        if not text or confidence < 0.5:
            await voice_service.speak_response("I didn't catch that. Could you please repeat?")
            return
        
        logger.info(f"Voice command: {text} (confidence: {confidence:.2f})")
        
        # Get current context
        context = context_manager.get_current_context()
        
        # Check if this is a long-running task
        if any(keyword in text.lower() for keyword in ["research", "find out", "report", "analyze"]):
            # Create async task
            task = await task_executor.create_task(text)
            response = "I'll work on that and notify you when I'm done."
        else:
            # Process immediately
            response, tool_results, thinking_time = await personal_agent.process_request(
                text, context
            )
        
        # Speak response
        await hardware_device.set_speaking(True)
        await voice_service.speak_response(response)
        await hardware_device.set_speaking(False)
        
    except Exception as e:
        logger.error(f"Error processing voice input: {e}")
        await voice_service.speak_response("I encountered an error processing your request.")
    finally:
        await hardware_device.set_processing(False)


async def proactive_monitoring():
    """Monitor for proactive assistance opportunities"""
    while True:
        try:
            # Get current context
            context = context_manager.get_current_context()
            
            # Check if we should interrupt
            if context_manager.should_interrupt():
                # Generate proactive suggestion
                suggestion = await personal_agent.generate_proactive_suggestion(context)
                
                if suggestion:
                    # Send notification to connected clients
                    message = WebSocketMessage(
                        type=MessageType.NOTIFICATION,
                        data={
                            "type": "proactive_suggestion",
                            "message": suggestion
                        }
                    )
                    await connection_manager.broadcast(message)
                    
                    # Optionally speak on hardware device
                    if hardware_device and not hardware_device.device_state["privacy_mode"]:
                        await voice_service.speak_response(suggestion)
            
            # Sleep for 5 minutes before next check
            await asyncio.sleep(300)
            
        except Exception as e:
            logger.error(f"Error in proactive monitoring: {e}")
            await asyncio.sleep(300)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, api_key: Optional[str] = None):
    """WebSocket endpoint for real-time communication"""
    # Validate API key
    if not await validate_websocket_key(websocket, api_key):
        return
    
    client_id = str(uuid.uuid4())
    await connection_manager.connect(websocket, client_id)
    
    # Send welcome message
    welcome_msg = WebSocketMessage(
        type=MessageType.STATUS,
        data={
            "status": "connected",
            "message": "Welcome to Wheatley 2.0!",
            "client_id": client_id
        }
    )
    await connection_manager.send_message(client_id, welcome_msg)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            message = WebSocketMessage(**data)
            
            # Handle different message types
            if message.type == MessageType.TEXT:
                # Process text message
                text_data = TextMessage(**message.data)
                context = context_manager.get_current_context()
                
                # Update context with interaction
                await context_manager.update_from_signals({"interaction": True})
                
                # Check if this is a long-running task
                if any(keyword in text_data.text.lower() for keyword in ["research", "find out", "report", "analyze"]):
                    # Create async task
                    task = await task_executor.create_task(text_data.text)
                    
                    response_msg = WebSocketMessage(
                        type=MessageType.RESPONSE,
                        data={
                            "text": "I'll work on that and notify you when I'm done.",
                            "task_id": task.id
                        }
                    )
                else:
                    # Process immediately
                    response_text, tool_results, thinking_time = await personal_agent.process_request(
                        text_data.text,
                        context
                    )
                    
                    response_msg = WebSocketMessage(
                        type=MessageType.RESPONSE,
                        data={
                            "text": response_text,
                            "tool_results": tool_results,
                            "thinking_time": thinking_time
                        }
                    )
                
                await connection_manager.send_message(client_id, response_msg)
            
            elif message.type == MessageType.AUDIO:
                # Process audio message
                audio_data = AudioMessage(**message.data)
                
                # Transcribe audio
                text, confidence = await voice_service.transcribe_audio(
                    audio_data.audio_data,
                    audio_data.sample_rate
                )
                
                if text and confidence > 0.5:
                    # Process as text
                    context = context_manager.get_current_context()
                    response_text, tool_results, thinking_time = await personal_agent.process_request(
                        text, context
                    )
                    
                    # Generate audio response
                    audio_response = await voice_service.synthesize_speech(response_text)
                    
                    response_msg = WebSocketMessage(
                        type=MessageType.RESPONSE,
                        data={
                            "text": response_text,
                            "audio": audio_response,
                            "transcription": text
                        }
                    )
                else:
                    response_msg = WebSocketMessage(
                        type=MessageType.ERROR,
                        data={"error": "Could not transcribe audio"}
                    )
                
                await connection_manager.send_message(client_id, response_msg)
            
            elif message.type == MessageType.COMMAND:
                # Handle system commands
                command = message.data.get("command")
                
                if command == "get_context":
                    context_data = context_manager.get_context_summary()
                    response_msg = WebSocketMessage(
                        type=MessageType.STATUS,
                        data=context_data
                    )
                elif command == "get_tasks":
                    active_tasks = task_executor.get_active_tasks()
                    completed_tasks = task_executor.get_completed_tasks()
                    response_msg = WebSocketMessage(
                        type=MessageType.STATUS,
                        data={
                            "active_tasks": [t.dict() for t in active_tasks],
                            "completed_tasks": [t.dict() for t in completed_tasks]
                        }
                    )
                else:
                    response_msg = WebSocketMessage(
                        type=MessageType.ERROR,
                        data={"error": f"Unknown command: {command}"}
                    )
                
                await connection_manager.send_message(client_id, response_msg)
                
    except WebSocketDisconnect:
        connection_manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        connection_manager.disconnect(client_id)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint - returns simple status page"""
    return """
    <html>
        <head>
            <title>Wheatley 2.0</title>
            <style>
                body { font-family: Arial, sans-serif; padding: 20px; }
                .status { color: green; }
            </style>
        </head>
        <body>
            <h1>Wheatley 2.0 - Personal AI Assistant</h1>
            <p class="status">Status: Online</p>
            <p>WebSocket endpoint: ws://localhost:8000/ws?api_key=YOUR_API_KEY</p>
            <p>For the web interface, please use the frontend application.</p>
        </body>
    </html>
    """


@app.get("/api/health")
async def health_check(api_key: str = Depends(get_api_key)):
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "agent": personal_agent is not None,
            "memory": memory_service is not None,
            "voice": voice_service is not None,
            "hardware": hardware_device is not None
        }
    }


@app.get("/api/tasks")
async def get_tasks(api_key: str = Depends(get_api_key)):
    """Get list of tasks"""
    active = task_executor.get_active_tasks()
    completed = task_executor.get_completed_tasks()
    
    return {
        "active": [task.dict() for task in active],
        "completed": [task.dict() for task in completed]
    }


@app.get("/api/context")
async def get_context(api_key: str = Depends(get_api_key)):
    """Get current context"""
    return context_manager.get_context_summary()


# Mount static files for frontend (if available)
frontend_path = Path("frontend")
if frontend_path.exists():
    app.mount("/app", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    ) 