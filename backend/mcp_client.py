import asyncio
import json
import os
from typing import Dict, List, Any, Optional
import httpx
from dataclasses import dataclass

@dataclass
class MCPServer:
    name: str
    command: str
    args: List[str]
    env: Optional[Dict[str, str]] = None

class MCPClient:
    """Basic MCP client implementation"""
    
    def __init__(self):
        self.servers: Dict[str, asyncio.subprocess.Process] = {}
        self.connections: Dict[str, Any] = {}
        self._request_id = 0
    
    async def connect_server(self, server: MCPServer):
        """Start and connect to an MCP server"""
        env = {**os.environ, **(server.env or {})}
        
        # Start the MCP server process
        process = await asyncio.create_subprocess_exec(
            server.command,
            *server.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        
        self.servers[server.name] = process
        
        # Initialize connection (simplified)
        await self._handshake(server.name, process)
        
    async def _handshake(self, name: str, process):
        """Perform MCP handshake"""
        # Send initialize request
        request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "clientInfo": {
                    "name": "wheatley",
                    "version": "2.0"
                }
            },
            "id": self._get_next_id()
        }
        
        process.stdin.write(json.dumps(request).encode() + b'\n')
        await process.stdin.drain()
        
        # Read response
        response = await process.stdout.readline()
        if response:
            self.connections[name] = json.loads(response)
        
    def _get_next_id(self):
        """Get next request ID"""
        self._request_id += 1
        return self._request_id
        
    async def call_tool(self, server_name: str, tool_name: str, args: Dict[str, Any]):
        """Call a tool on an MCP server"""
        if server_name not in self.servers:
            raise ValueError(f"Server {server_name} not connected")
        
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": args
            },
            "id": self._get_next_id()
        }
        
        process = self.servers[server_name]
        process.stdin.write(json.dumps(request).encode() + b'\n')
        await process.stdin.drain()
        
        response = await process.stdout.readline()
        if response:
            return json.loads(response)
        return None
    
    async def list_tools(self, server_name: str):
        """List available tools on an MCP server"""
        if server_name not in self.servers:
            raise ValueError(f"Server {server_name} not connected")
        
        request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": self._get_next_id()
        }
        
        process = self.servers[server_name]
        process.stdin.write(json.dumps(request).encode() + b'\n')
        await process.stdin.drain()
        
        response = await process.stdout.readline()
        if response:
            return json.loads(response)
        return None
    
    async def disconnect_all(self):
        """Disconnect all MCP servers"""
        for name, process in self.servers.items():
            try:
                process.terminate()
                await process.wait()
            except Exception as e:
                print(f"Error disconnecting {name}: {e}")
        self.servers.clear()
        self.connections.clear()