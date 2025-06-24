import asyncio
import uuid
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import anthropic
import google.generativeai as genai
from mcp_client import MCPClient, MCPServer
from config import settings

@dataclass
class AgentState:
    task: str
    status: str = "initializing"
    memory: List[Dict[str, Any]] = field(default_factory=list)
    sandbox_path: str = ""
    mcp_servers: List[str] = field(default_factory=list)
    model: str = "claude-4-opus"
    created_at: datetime = field(default_factory=datetime.utcnow)

class Agent:
    def __init__(self, task: str, model: str = "claude-4-opus"):
        self.id = str(uuid.uuid4())
        self.state = AgentState(task=task, model=model)
        self.mcp_client = MCPClient()
        self._setup_sandbox()
        
    def _setup_sandbox(self):
        """Create agent sandbox directory"""
        self.state.sandbox_path = f"{settings.sandbox_path}/agents/{self.id}"
        os.makedirs(self.state.sandbox_path, exist_ok=True)
        
    async def add_mcp_server(self, server_name: str):
        """Connect to an MCP server"""
        # Define available MCP servers
        servers = {
            "filesystem": MCPServer(
                name="filesystem",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", self.state.sandbox_path]
            ),
            "github": MCPServer(
                name="github",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-github"],
                env={"GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_TOKEN", "")}
            ),
            "fetch": MCPServer(
                name="fetch",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-fetch"]
            ),
        }
        
        if server_name in servers:
            try:
                await self.mcp_client.connect_server(servers[server_name])
                self.state.mcp_servers.append(server_name)
            except Exception as e:
                print(f"Failed to connect to {server_name}: {e}")
    
    async def think(self) -> Dict[str, Any]:
        """Agent thinks about what to do next"""
        self.state.status = "thinking"
        
        # Prepare context from memory
        context = "\n".join([
            f"{m['type']}: {m['content']}" 
            for m in self.state.memory[-10:]  # Last 10 memories
        ])
        
        prompt = f"""
        Task: {self.state.task}
        
        Current context:
        {context}
        
        Available MCP servers: {', '.join(self.state.mcp_servers)}
        
        What should I do next to complete this task? 
        If you need to use a tool, specify which MCP server and tool.
        Be specific about the next action to take.
        """
        
        if self.state.model.startswith("claude"):
            client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
            response = await client.messages.create(
                model=self.state.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            thought = response.content[0].text
        else:
            # Use Gemini
            genai.configure(api_key=settings.gemini_api_key)
            model = genai.GenerativeModel(self.state.model)
            response = await model.generate_content_async(prompt)
            thought = response.text
        
        return {
            "thought": thought,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def act(self, thought: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action based on thought"""
        self.state.status = "acting"
        
        # Parse thought for tool usage
        thought_text = thought["thought"]
        
        # Simple parsing for MCP commands
        if "mcp:" in thought_text.lower():
            try:
                # Extract MCP command (simplified parsing)
                lines = thought_text.split('\n')
                for line in lines:
                    if "mcp:" in line.lower():
                        parts = line.split("mcp:")[1].strip().split()
                        if len(parts) >= 2:
                            server_name = parts[0]
                            tool_name = parts[1]
                            args = {}
                            if len(parts) > 2:
                                try:
                                    args = json.loads(' '.join(parts[2:]))
                                except json.JSONDecodeError:
                                    args = {"query": ' '.join(parts[2:])}
                            
                            result = await self.mcp_client.call_tool(server_name, tool_name, args)
                            return {
                                "action": "tool_call",
                                "server": server_name,
                                "tool": tool_name,
                                "args": args,
                                "result": result
                            }
            except Exception as e:
                return {
                    "action": "error",
                    "error": f"Failed to execute tool: {str(e)}"
                }
        
        # No tool needed, just thinking
        return {
            "action": "continue",
            "reasoning": thought_text
        }
    
    async def observe(self, action_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process action results and update memory"""
        self.state.status = "observing"
        
        observation = {
            "type": "observation",
            "content": action_result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.state.memory.append(observation)
        
        # Check if task is complete
        if self._is_task_complete():
            self.state.status = "completed"
            return {"status": "completed", "result": self._summarize_results()}
        
        return {"status": "continue"}
    
    def _is_task_complete(self) -> bool:
        """Determine if the task is complete"""
        # Prevent infinite loops
        if len(self.state.memory) > 50:
            return True
        
        # Check if recent observations indicate completion
        recent_observations = [m for m in self.state.memory[-5:] if m.get("type") == "observation"]
        for obs in recent_observations:
            content_str = str(obs.get("content", "")).lower()
            if any(keyword in content_str for keyword in ["completed", "finished", "done", "success"]):
                return True
        
        return False
    
    def _summarize_results(self) -> str:
        """Summarize the agent's work"""
        return f"Task '{self.state.task}' completed with {len(self.state.memory)} steps."
    
    async def execute(self) -> Dict[str, Any]:
        """Main execution loop"""
        self.state.status = "running"
        
        while self.state.status not in ["completed", "failed"]:
            try:
                # Think
                thought = await self.think()
                self.state.memory.append({"type": "thought", "content": thought})
                
                # Act
                action = await self.act(thought)
                self.state.memory.append({"type": "action", "content": action})
                
                # Observe
                observation = await self.observe(action)
                
                if observation["status"] == "completed":
                    break
                    
                # Small delay to prevent spinning
                await asyncio.sleep(0.5)
                
            except Exception as e:
                self.state.status = "failed"
                return {
                    "status": "failed",
                    "error": str(e),
                    "memory": self.state.memory
                }
        
        return {
            "status": self.state.status,
            "result": observation.get("result", ""),
            "steps": len(self.state.memory),
            "memory": self.state.memory[-10:]  # Last 10 items
        }

class AgentManager:
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.active_tasks: Dict[str, asyncio.Task] = {}
    
    async def create_agent(self, task: str, model: str = "claude-4-opus") -> str:
        """Create and start a new agent"""
        agent = Agent(task, model)
        self.agents[agent.id] = agent
        
        # Start agent execution in background
        task_coroutine = agent.execute()
        self.active_tasks[agent.id] = asyncio.create_task(task_coroutine)
        
        return agent.id
    
    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get current status of an agent"""
        if agent_id not in self.agents:
            return {"error": "Agent not found"}
        
        agent = self.agents[agent_id]
        return {
            "id": agent_id,
            "task": agent.state.task,
            "status": agent.state.status,
            "memory_size": len(agent.state.memory),
            "mcp_servers": agent.state.mcp_servers,
            "created_at": agent.state.created_at.isoformat()
        }
    
    async def stop_agent(self, agent_id: str):
        """Stop a running agent"""
        if agent_id in self.active_tasks:
            self.active_tasks[agent_id].cancel()
            del self.active_tasks[agent_id]
        
        if agent_id in self.agents:
            await self.agents[agent_id].mcp_client.disconnect_all()
            del self.agents[agent_id]