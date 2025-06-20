"""
Personal Agent service using Gemini 2.5 Flash
"""
import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from backend.config import settings, user_config
from backend.models.schemas import (
    UserContext, Memory, ToolCall, Conversation
)
from backend.services.memory import MemoryService
from backend.services.mcp_tools import MCPToolManager

logger = logging.getLogger(__name__)


class PersonalAgent:
    """Personal AI agent powered by Gemini 2.5 Flash"""
    
    def __init__(self, memory_service: MemoryService, tool_manager: MCPToolManager):
        self.memory_service = memory_service
        self.tool_manager = tool_manager
        
        # Configure Gemini
        genai.configure(api_key=settings.gemini_api_key)
        
        # Initialize model with thinking budget
        self.model = genai.GenerativeModel(
            model_name=settings.gemini_model,
            generation_config=genai.GenerationConfig(
                temperature=settings.gemini_temperature,
                max_output_tokens=settings.gemini_max_tokens,
            ),
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        # Initialize search-grounded model for web searches
        self.search_model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config=genai.GenerationConfig(
                temperature=0.7,
                max_output_tokens=1024,
            ),
            tools=[
                genai.Tool(
                    function_declarations=[
                        genai.FunctionDeclaration(
                            name="google_search",
                            description="Search the web for current information",
                            parameters={
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "The search query"
                                    }
                                },
                                "required": ["query"]
                            }
                        )
                    ]
                )
            ]
        )
        
        self.conversation_history: List[Dict[str, str]] = []
        self.current_context: Optional[UserContext] = None
        
    def _build_system_prompt(self, context: UserContext) -> str:
        """Build personalized system prompt based on context and preferences"""
        personality = user_config["assistant"]["personality"]
        interests = user_config["preferences"]["interests"]
        communication = user_config["preferences"]["communication"]
        
        prompt = f"""You are {user_config['assistant']['name']}, a {personality} personal AI assistant.

Current Context:
- Time: {context.current_time.strftime('%Y-%m-%d %H:%M:%S')}
- Location: {context.location}
- User Activity: {context.activity}
- User Mood: {context.mood}

User Preferences:
- Interests: {', '.join(interests)}
- Communication Style: {communication['formality']}, {communication['verbosity']}
- Humor: {communication['humor']}

Guidelines:
1. Be {personality} in all interactions
2. Keep responses {communication['verbosity']}
3. Use {communication['formality']} language
4. Remember you're a personal assistant who knows the user well
5. Be proactive and suggest helpful actions based on context
6. Use available tools when appropriate

Available Tools:
{self.tool_manager.get_available_tools_description()}

When you need to use a tool, respond with a JSON block like this:
```json
{{
    "tool": "tool_name",
    "parameters": {{
        "param1": "value1",
        "param2": "value2"
    }}
}}
```

After using tools, provide a natural response incorporating the results."""
        
        return prompt
    
    async def process_request(
        self,
        text: str,
        context: UserContext,
        include_memory: bool = True
    ) -> Tuple[str, Dict[str, Any], float]:
        """
        Process a user request with full context
        
        Returns:
            Tuple of (response_text, tool_results, thinking_time)
        """
        start_time = time.time()
        self.current_context = context
        
        # Retrieve relevant memories
        relevant_memories = []
        if include_memory:
            memories = await self.memory_service.search_memories(text, limit=5)
            relevant_memories = [m.content for m in memories]
        
        # Build the full prompt
        system_prompt = self._build_system_prompt(context)
        
        # Add memory context if available
        memory_context = ""
        if relevant_memories:
            memory_context = "\n\nRelevant memories:\n" + "\n".join(
                f"- {mem}" for mem in relevant_memories
            )
        
        # Prepare the conversation
        messages = [
            {"role": "user", "content": system_prompt + memory_context},
            {"role": "user", "content": text}
        ]
        
        # Add recent conversation history
        for msg in self.conversation_history[-5:]:  # Last 5 exchanges
            messages.append(msg)
        
        tool_results = {}
        response_text = ""
        
        try:
            # Generate response with thinking budget
            response = self.model.generate_content(
                messages,
                generation_config=genai.GenerationConfig(
                    temperature=settings.gemini_temperature,
                    max_output_tokens=settings.gemini_max_tokens,
                    # Note: Thinking budget would be applied here if supported
                )
            )
            
            response_text = response.text
            
            # Check for tool usage in response
            tool_calls = self._extract_tool_calls(response_text)
            
            if tool_calls:
                # Execute tools
                for tool_call in tool_calls:
                    result = await self.tool_manager.execute_tool(
                        tool_call["tool"],
                        tool_call["parameters"]
                    )
                    tool_results[tool_call["tool"]] = result
                
                # Generate final response with tool results
                tool_response = self._format_tool_results(tool_results)
                final_prompt = f"{text}\n\nTool results:\n{tool_response}\n\nPlease provide a natural response incorporating this information."
                
                final_response = self.model.generate_content(final_prompt)
                response_text = final_response.text
            
            # Store in conversation history
            self.conversation_history.append({"role": "user", "content": text})
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            # Keep history manageable
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            # Store important information in memory
            await self._store_important_info(text, response_text, context)
            
        except Exception as e:
            logger.error(f"Error in agent processing: {e}")
            response_text = "I apologize, but I encountered an error processing your request."
        
        thinking_time = time.time() - start_time
        return response_text, tool_results, thinking_time
    
    async def search_web(self, query: str) -> str:
        """Use Gemini with search grounding for web searches"""
        try:
            # Use the search-grounded model
            response = self.search_model.generate_content(
                f"Search for current information about: {query}",
                tools='google_search_retrieval'  # Enable search grounding
            )
            
            return response.text
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return f"I couldn't search for that information right now."
    
    def _extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Extract tool calls from response text"""
        tool_calls = []
        
        # Look for JSON blocks in the response
        import re
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                tool_data = json.loads(match)
                if "tool" in tool_data and "parameters" in tool_data:
                    tool_calls.append(tool_data)
            except json.JSONDecodeError:
                continue
        
        return tool_calls
    
    def _format_tool_results(self, results: Dict[str, Any]) -> str:
        """Format tool results for inclusion in prompt"""
        formatted = []
        for tool, result in results.items():
            if isinstance(result, dict):
                formatted.append(f"{tool}: {json.dumps(result, indent=2)}")
            else:
                formatted.append(f"{tool}: {result}")
        return "\n".join(formatted)
    
    async def _store_important_info(
        self,
        user_input: str,
        response: str,
        context: UserContext
    ):
        """Identify and store important information in memory"""
        # Simple importance detection (can be enhanced)
        important_keywords = [
            "remember", "my favorite", "i prefer", "always", "never",
            "i am", "my name", "birthday", "anniversary"
        ]
        
        is_important = any(
            keyword in user_input.lower()
            for keyword in important_keywords
        )
        
        if is_important:
            memory_content = f"User said: {user_input}\nContext: {context.activity} at {context.location}"
            await self.memory_service.store_memory(
                content=memory_content,
                context=context.dict(),
                importance=0.8
            )
    
    async def generate_proactive_suggestion(
        self,
        context: UserContext
    ) -> Optional[str]:
        """Generate proactive suggestions based on context"""
        current_hour = context.current_time.hour
        
        # Morning briefing
        briefing_hour = int(user_config["preferences"]["morning_briefing_time"].split(":")[0])
        if current_hour == briefing_hour and context.activity != "morning_briefing":
            return await self._generate_morning_briefing(context)
        
        # Meal reminders
        if context.location == "kitchen" and current_hour in [12, 18]:
            return "Would you like some recipe suggestions for lunch/dinner?"
        
        # Focus time reminders
        if 10 <= current_hour <= 16 and context.activity == "working":
            last_break = context.environment.get("last_break_time")
            if last_break and (context.current_time - last_break).seconds > 7200:  # 2 hours
                return "You've been working for a while. How about a short break?"
        
        return None
    
    async def _generate_morning_briefing(self, context: UserContext) -> str:
        """Generate personalized morning briefing"""
        routines = user_config["routines"]["morning"]
        briefing_parts = []
        
        for routine in routines:
            if routine == "weather":
                # Use weather tool
                weather = await self.tool_manager.execute_tool(
                    "weather",
                    {"location": user_config["preferences"]["weather_location"]}
                )
                briefing_parts.append(f"Weather: {weather}")
            
            elif routine == "calendar":
                # Use calendar tool
                events = await self.tool_manager.execute_tool(
                    "get_today_events",
                    {}
                )
                briefing_parts.append(f"Today's events: {events}")
            
            elif routine == "news_summary":
                # Get news summary
                news = await self.search_web(
                    f"latest news {' '.join(user_config['preferences']['news_sources'])}"
                )
                briefing_parts.append(f"News: {news[:200]}...")
        
        briefing = "Good morning! Here's your briefing:\n" + "\n".join(briefing_parts)
        return briefing 