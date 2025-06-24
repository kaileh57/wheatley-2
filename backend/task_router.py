import google.generativeai as genai
from typing import Dict, Any
import asyncio
import os
from agent_manager import AgentManager
from config import settings

class TaskRouter:
    def __init__(self, agent_manager: AgentManager):
        self.agent_manager = agent_manager
        genai.configure(api_key=settings.gemini_api_key)
        
    async def route_task(self, query: str) -> Dict[str, Any]:
        """Determine if task is simple or complex and route accordingly"""
        
        # Use Gemini to classify the task
        classification_prompt = f"""
        Classify this query as either "simple" (can be answered in <10 seconds with search) 
        or "complex" (requires multiple steps, tools, or deep analysis):
        
        Query: {query}
        
        Respond with just "simple" or "complex" and a brief reason.
        """
        
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = await model.generate_content_async(classification_prompt)
        
        classification = response.text.lower()
        
        if "simple" in classification:
            return await self._handle_simple_query(query)
        else:
            return await self._handle_complex_task(query)
    
    async def _handle_simple_query(self, query: str) -> Dict[str, Any]:
        """Handle simple queries with Gemini + Google Search grounding"""
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        try:
            # Enable search grounding
            response = await model.generate_content_async(
                query,
                tools=[{'google_search_retrieval': {}}],
                tool_config={
                    'google_search_retrieval': {
                        'dynamic_retrieval_config': {
                            'mode': 'MODE_DYNAMIC',
                            'dynamic_threshold': 0.3
                        }
                    }
                }
            )
            
            return {
                "type": "simple",
                "response": response.text,
                "grounding_metadata": getattr(response, 'grounding_metadata', None)
            }
        except Exception as e:
            # Fallback to simple generation without search
            response = await model.generate_content_async(query)
            return {
                "type": "simple",
                "response": response.text,
                "note": f"Search grounding failed: {str(e)}"
            }
    
    async def _handle_complex_task(self, query: str) -> Dict[str, Any]:
        """Handle complex tasks by creating an agent"""
        
        # Determine which MCP servers might be needed
        mcp_analysis = await self._analyze_required_mcps(query)
        
        # Create agent
        agent_id = await self.agent_manager.create_agent(query)
        
        # Add required MCP servers
        agent = self.agent_manager.agents[agent_id]
        for server in mcp_analysis.get("servers", []):
            await agent.add_mcp_server(server)
        
        return {
            "type": "complex",
            "agent_id": agent_id,
            "message": f"Agent {agent_id} created and working on your task",
            "mcp_servers": mcp_analysis.get("servers", [])
        }
    
    async def _analyze_required_mcps(self, query: str) -> Dict[str, Any]:
        """Analyze which MCP servers might be needed"""
        analysis_prompt = f"""
        Given this task, which MCP servers would be helpful?
        Available servers: filesystem, github, fetch
        
        Task: {query}
        
        List only the server names that would be useful, separated by commas.
        """
        
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = await model.generate_content_async(analysis_prompt)
        
        # Parse response for server names
        servers = []
        available_servers = ["filesystem", "github", "fetch"]
        for server in available_servers:
            if server in response.text.lower():
                servers.append(server)
        
        return {"servers": servers}