import asyncio
import json
import sys
import os
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
from dotenv import load_dotenv
import httpx

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerplexityClient:
    """Client for interacting with Perplexity API"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.perplexity.ai"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers=self.headers,
            timeout=300
        )
    
    async def search(self, query: str, **kwargs) -> Dict[str, Any]:
        """Perform a search using Perplexity's API"""
        payload = {
            "model": "sonar-pro",
            "messages": [
                {"role": "system", "content": "Be precise and comprehensive in your research."},
                {"role": "user", "content": query}
            ],
            "temperature": kwargs.get("temperature", 0.2),
            "return_images": kwargs.get("return_images", False),
            "return_related_questions": kwargs.get("return_related_questions", True)
        }
        
        if kwargs.get("search_domain_filter"):
            payload["search_domain_filter"] = kwargs["search_domain_filter"]
        if kwargs.get("search_recency_filter"):
            payload["search_recency_filter"] = kwargs["search_recency_filter"]
        
        response = await self.client.post("/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

class PerplexityMCPServer:
    """MCP Server for Perplexity integration"""
    
    def __init__(self):
        self.request_id = 0
        self.perplexity_client = None
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize Perplexity client"""
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            raise ValueError("PERPLEXITY_API_KEY environment variable is required")
        
        self.perplexity_client = PerplexityClient(
            api_key=api_key,
            base_url=os.getenv("PERPLEXITY_BASE_URL", "https://api.perplexity.ai")
        )
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming JSON-RPC request"""
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")
        
        logger.debug(f"Handling request: {method}")
        
        try:
            if method == "initialize":
                return await self._handle_initialize(request_id, params)
            elif method == "tools/list":
                return await self._handle_list_tools(request_id)
            elif method == "tools/call":
                return await self._handle_tool_call(request_id, params)
            else:
                return self._create_error_response(
                    request_id,
                    -32601,
                    f"Method not found: {method}"
                )
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            return self._create_error_response(
                request_id,
                -32603,
                f"Internal error: {str(e)}"
            )
    
    async def _handle_initialize(self, request_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialize request"""
        client_info = params.get("clientInfo", {})
        logger.info(f"Initializing connection with client: {client_info.get('name', 'unknown')}")
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "perplexity-mcp-server",
                    "version": "1.0.0"
                }
            }
        }
    
    async def _handle_list_tools(self, request_id: Any) -> Dict[str, Any]:
        """Handle tools/list request"""
        tools = [
            {
                "name": "perplexity_search",
                "description": "Perform a web search using Perplexity's AI-powered search",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "search_domain_filter": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional: Limit search to specific domains"
                        },
                        "search_recency_filter": {
                            "type": "string",
                            "enum": ["day", "week", "month", "year"],
                            "description": "Optional: Filter by recency"
                        },
                        "return_images": {
                            "type": "boolean",
                            "default": False,
                            "description": "Whether to return images"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": tools
            }
        }
    
    async def _handle_tool_call(self, request_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        logger.info(f"Calling tool: {tool_name}")
        
        if tool_name != "perplexity_search":
            return self._create_error_response(
                request_id,
                -32602,
                f"Unknown tool: {tool_name}"
            )
        
        try:
            result = await self.perplexity_client.search(
                query=arguments["query"],
                search_domain_filter=arguments.get("search_domain_filter"),
                search_recency_filter=arguments.get("search_recency_filter"),
                return_images=arguments.get("return_images", False)
            )
            
            # Format response
            formatted_result = {
                "query": arguments["query"],
                "answer": result["choices"][0]["message"]["content"],
                "citations": result.get("citations", []),
                "related_questions": result.get("related_questions", [])
            }
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(formatted_result, indent=2)
                        }
                    ]
                }
            }
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return self._create_error_response(
                request_id,
                -32603,
                f"Tool execution failed: {str(e)}"
            )
    
    def _create_error_response(self, request_id: Any, code: int, message: str) -> Dict[str, Any]:
        """Create JSON-RPC error response"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }
    
    async def run(self):
        """Main server loop"""
        logger.info("Perplexity MCP Server starting...")
        
        # Read from stdin and write to stdout
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_running_loop().connect_read_pipe(
            lambda: protocol, sys.stdin
        )
        
        while True:
            try:
                # Read line from stdin
                line = await reader.readline()
                if not line:
                    break
                
                # Parse JSON-RPC request
                request = json.loads(line.decode())
                logger.debug(f"Received request: {request}")
                
                # Handle request
                response = await self.handle_request(request)
                
                # Send response
                sys.stdout.write(json.dumps(response) + '\n')
                sys.stdout.flush()
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
                error_response = self._create_error_response(
                    None,
                    -32700,
                    "Parse error"
                )
                sys.stdout.write(json.dumps(error_response) + '\n')
                sys.stdout.flush()
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                break
        
        # Cleanup
        if self.perplexity_client:
            await self.perplexity_client.close()

async def main():
    """Main entry point"""
    server = PerplexityMCPServer()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())