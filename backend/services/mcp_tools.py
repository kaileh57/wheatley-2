"""
MCP (Model Context Protocol) Tool Manager
Manages and executes various tools for the personal assistant
"""
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import aiohttp
from pathlib import Path

from backend.config import settings

logger = logging.getLogger(__name__)


class Tool:
    """Base class for all tools"""
    
    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters
    
    async def execute(self, params: Dict[str, Any]) -> Any:
        """Execute the tool with given parameters"""
        raise NotImplementedError


class WeatherTool(Tool):
    """Get weather information for a location"""
    
    def __init__(self):
        super().__init__(
            name="weather",
            description="Get current weather for a location",
            parameters={
                "location": {
                    "type": "string",
                    "description": "City name or location",
                    "required": True
                }
            }
        )
    
    async def execute(self, params: Dict[str, Any]) -> Any:
        location = params.get("location")
        if not location:
            return {"error": "Location is required"}
        
        # For demo purposes, return mock data
        # In production, integrate with weather API
        return {
            "location": location,
            "temperature": "72°F",
            "condition": "Partly cloudy",
            "humidity": "65%",
            "wind": "10 mph"
        }


class CalendarTool(Tool):
    """Manage calendar events"""
    
    def __init__(self):
        super().__init__(
            name="calendar",
            description="Create, read, update calendar events",
            parameters={
                "action": {
                    "type": "string",
                    "enum": ["create", "list", "update", "delete"],
                    "required": True
                },
                "event_data": {
                    "type": "object",
                    "description": "Event details",
                    "required": False
                }
            }
        )
        self.events_file = Path("data/calendar_events.json")
        self.events_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_events()
    
    def _load_events(self):
        """Load events from file"""
        if self.events_file.exists():
            with open(self.events_file, 'r') as f:
                self.events = json.load(f)
        else:
            self.events = []
    
    def _save_events(self):
        """Save events to file"""
        with open(self.events_file, 'w') as f:
            json.dump(self.events, f, indent=2)
    
    async def execute(self, params: Dict[str, Any]) -> Any:
        action = params.get("action")
        event_data = params.get("event_data", {})
        
        if action == "create":
            event = {
                "id": len(self.events) + 1,
                "title": event_data.get("title", "Untitled Event"),
                "date": event_data.get("date", datetime.now().isoformat()),
                "description": event_data.get("description", ""),
                "created_at": datetime.now().isoformat()
            }
            self.events.append(event)
            self._save_events()
            return {"success": True, "event": event}
        
        elif action == "list":
            # Get today's events by default
            today = datetime.now().date()
            today_events = [
                e for e in self.events
                if datetime.fromisoformat(e["date"]).date() == today
            ]
            return {"events": today_events}
        
        return {"error": f"Unsupported action: {action}"}


class TimerTool(Tool):
    """Set timers and alarms"""
    
    def __init__(self):
        super().__init__(
            name="timer",
            description="Set a timer for a specified duration",
            parameters={
                "duration": {
                    "type": "string",
                    "description": "Duration (e.g., '5m', '1h', '30s')",
                    "required": True
                },
                "message": {
                    "type": "string",
                    "description": "Message to show when timer expires",
                    "required": False
                }
            }
        )
        self.active_timers = []
    
    async def execute(self, params: Dict[str, Any]) -> Any:
        duration_str = params.get("duration")
        message = params.get("message", "Timer expired!")
        
        # Parse duration
        duration_seconds = self._parse_duration(duration_str)
        if duration_seconds is None:
            return {"error": "Invalid duration format"}
        
        # Create timer
        timer_id = len(self.active_timers) + 1
        timer_data = {
            "id": timer_id,
            "duration": duration_seconds,
            "message": message,
            "created_at": datetime.now().isoformat()
        }
        
        self.active_timers.append(timer_data)
        
        # Start timer in background
        asyncio.create_task(self._run_timer(timer_id, duration_seconds, message))
        
        return {
            "success": True,
            "timer_id": timer_id,
            "duration": duration_str,
            "message": message
        }
    
    def _parse_duration(self, duration_str: str) -> Optional[int]:
        """Parse duration string to seconds"""
        try:
            if duration_str.endswith('s'):
                return int(duration_str[:-1])
            elif duration_str.endswith('m'):
                return int(duration_str[:-1]) * 60
            elif duration_str.endswith('h'):
                return int(duration_str[:-1]) * 3600
            else:
                return None
        except ValueError:
            return None
    
    async def _run_timer(self, timer_id: int, duration: int, message: str):
        """Run timer in background"""
        await asyncio.sleep(duration)
        logger.info(f"Timer {timer_id} expired: {message}")
        # In real implementation, this would trigger a notification


class ReminderTool(Tool):
    """Set reminders"""
    
    def __init__(self):
        super().__init__(
            name="reminder",
            description="Set a reminder for a specific time",
            parameters={
                "time": {
                    "type": "string",
                    "description": "Time for reminder (HH:MM or ISO format)",
                    "required": True
                },
                "message": {
                    "type": "string",
                    "description": "Reminder message",
                    "required": True
                },
                "recurring": {
                    "type": "boolean",
                    "description": "Whether this is a recurring reminder",
                    "required": False
                }
            }
        )
        self.reminders_file = Path("data/reminders.json")
        self.reminders_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_reminders()
    
    def _load_reminders(self):
        """Load reminders from file"""
        if self.reminders_file.exists():
            with open(self.reminders_file, 'r') as f:
                self.reminders = json.load(f)
        else:
            self.reminders = []
    
    def _save_reminders(self):
        """Save reminders to file"""
        with open(self.reminders_file, 'w') as f:
            json.dump(self.reminders, f, indent=2)
    
    async def execute(self, params: Dict[str, Any]) -> Any:
        time_str = params.get("time")
        message = params.get("message")
        recurring = params.get("recurring", False)
        
        if not time_str or not message:
            return {"error": "Time and message are required"}
        
        # Parse time
        try:
            if ":" in time_str and len(time_str) <= 5:
                # Simple HH:MM format
                hour, minute = map(int, time_str.split(":"))
                reminder_time = datetime.now().replace(
                    hour=hour, minute=minute, second=0, microsecond=0
                )
                if reminder_time < datetime.now():
                    reminder_time += timedelta(days=1)
            else:
                # ISO format
                reminder_time = datetime.fromisoformat(time_str)
        except ValueError:
            return {"error": "Invalid time format"}
        
        reminder = {
            "id": len(self.reminders) + 1,
            "time": reminder_time.isoformat(),
            "message": message,
            "recurring": recurring,
            "created_at": datetime.now().isoformat(),
            "active": True
        }
        
        self.reminders.append(reminder)
        self._save_reminders()
        
        return {
            "success": True,
            "reminder": reminder,
            "scheduled_for": reminder_time.strftime("%Y-%m-%d %H:%M")
        }


class NoteTool(Tool):
    """Take and manage notes"""
    
    def __init__(self):
        super().__init__(
            name="note",
            description="Create, read, and manage notes",
            parameters={
                "action": {
                    "type": "string",
                    "enum": ["create", "list", "search"],
                    "required": True
                },
                "content": {
                    "type": "string",
                    "description": "Note content",
                    "required": False
                },
                "query": {
                    "type": "string",
                    "description": "Search query",
                    "required": False
                }
            }
        )
        self.notes_file = Path("data/notes.json")
        self.notes_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_notes()
    
    def _load_notes(self):
        """Load notes from file"""
        if self.notes_file.exists():
            with open(self.notes_file, 'r') as f:
                self.notes = json.load(f)
        else:
            self.notes = []
    
    def _save_notes(self):
        """Save notes to file"""
        with open(self.notes_file, 'w') as f:
            json.dump(self.notes, f, indent=2)
    
    async def execute(self, params: Dict[str, Any]) -> Any:
        action = params.get("action")
        
        if action == "create":
            content = params.get("content")
            if not content:
                return {"error": "Content is required for creating a note"}
            
            note = {
                "id": len(self.notes) + 1,
                "content": content,
                "created_at": datetime.now().isoformat(),
                "tags": self._extract_tags(content)
            }
            
            self.notes.append(note)
            self._save_notes()
            
            return {"success": True, "note": note}
        
        elif action == "list":
            # Return recent notes
            recent_notes = sorted(
                self.notes,
                key=lambda x: x["created_at"],
                reverse=True
            )[:10]
            return {"notes": recent_notes}
        
        elif action == "search":
            query = params.get("query", "").lower()
            matching_notes = [
                note for note in self.notes
                if query in note["content"].lower()
            ]
            return {"notes": matching_notes}
        
        return {"error": f"Unsupported action: {action}"}
    
    def _extract_tags(self, content: str) -> List[str]:
        """Extract hashtags from content"""
        import re
        tags = re.findall(r'#(\w+)', content)
        return list(set(tags))


class MCPToolManager:
    """Manages all available tools"""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._register_default_tools()
        logger.info(f"Initialized MCP Tool Manager with {len(self.tools)} tools")
    
    def _register_default_tools(self):
        """Register default tools"""
        default_tools = [
            WeatherTool(),
            CalendarTool(),
            TimerTool(),
            ReminderTool(),
            NoteTool()
        ]
        
        for tool in default_tools:
            self.register_tool(tool)
    
    def register_tool(self, tool: Tool):
        """Register a new tool"""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names"""
        return list(self.tools.keys())
    
    def get_available_tools_description(self) -> str:
        """Get formatted description of all available tools"""
        descriptions = []
        for name, tool in self.tools.items():
            param_desc = []
            for param_name, param_info in tool.parameters.items():
                required = param_info.get("required", False)
                param_type = param_info.get("type", "any")
                desc = param_info.get("description", "")
                param_desc.append(
                    f"  - {param_name} ({param_type}{'*' if required else ''}): {desc}"
                )
            
            descriptions.append(
                f"- {name}: {tool.description}\n" + "\n".join(param_desc)
            )
        
        return "\n\n".join(descriptions)
    
    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> Any:
        """Execute a tool with given parameters"""
        if tool_name not in self.tools:
            return {"error": f"Unknown tool: {tool_name}"}
        
        tool = self.tools[tool_name]
        
        # Validate required parameters
        for param_name, param_info in tool.parameters.items():
            if param_info.get("required", False) and param_name not in parameters:
                return {"error": f"Missing required parameter: {param_name}"}
        
        try:
            result = await tool.execute(parameters)
            logger.info(f"Executed tool {tool_name} successfully")
            return result
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {"error": f"Tool execution failed: {str(e)}"} 