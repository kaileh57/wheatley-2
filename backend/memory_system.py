import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
import openai
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import Session
from models import Memory, UserProfile
import asyncio

class AdvancedMemorySystem:
    def __init__(self, db_session: Session, openai_api_key: str):
        self.db = db_session
        self.openai_client = openai.AsyncOpenAI(api_key=openai_api_key)
        
    async def embed_text(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI"""
        response = await self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    
    async def learn_from_conversation(self, messages: List[Dict[str, str]]):
        """Extract and store important information from conversations"""
        # Analyze conversation for important facts
        analysis_prompt = f"""
        Analyze this conversation and extract:
        1. Important facts about the user
        2. User preferences and habits
        3. Significant events or plans mentioned
        4. Technical details or requirements
        
        Conversation:
        {json.dumps(messages, indent=2)}
        
        Return as JSON with categories: facts, preferences, events, technical
        """
        
        # Use Claude for analysis
        import anthropic
        from config import settings
        claude = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        response = await claude.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": analysis_prompt}]
        )
        
        try:
            extracted = json.loads(response.content[0].text)
            
            # Store facts in user profile
            for fact in extracted.get('facts', []):
                self._update_user_profile('fact', fact)
            
            # Store preferences
            for pref in extracted.get('preferences', []):
                self._update_user_profile('preference', pref)
            
            # Create memories for important items
            for category, items in extracted.items():
                if isinstance(items, list):
                    for item in items:
                        await self._store_memory(item, category)
                        
        except (json.JSONDecodeError, Exception) as e:
            # Fallback: store entire conversation as memory
            conversation_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            await self._store_memory(conversation_text, "conversation")
    
    async def _store_memory(self, content: str, category: str, importance: int = 5):
        """Store a memory with embedding"""
        embedding = await self.embed_text(content)
        
        memory = Memory(
            content=content,
            embedding=json.dumps(embedding),
            category=category,
            importance=importance
        )
        self.db.add(memory)
        self.db.commit()
    
    def _update_user_profile(self, key: str, value: str):
        """Update user profile information"""
        profile_entry = self.db.query(UserProfile).filter_by(key=key).first()
        
        if profile_entry:
            # Append to existing value
            try:
                existing = json.loads(profile_entry.value) if profile_entry.value.startswith('[') else [profile_entry.value]
            except json.JSONDecodeError:
                existing = [profile_entry.value]
            existing.append(value)
            profile_entry.value = json.dumps(existing)
            profile_entry.updated_at = datetime.utcnow()
        else:
            # Create new entry
            profile_entry = UserProfile(key=key, value=json.dumps([value]))
            self.db.add(profile_entry)
        
        self.db.commit()
    
    async def get_relevant_context(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Retrieve relevant memories for a query"""
        query_embedding = await self.embed_text(query)
        
        # Get all memories
        memories = self.db.query(Memory).all()
        
        # Calculate similarities
        similarities = []
        for memory in memories:
            try:
                mem_embedding = json.loads(memory.embedding)
                similarity = cosine_similarity([query_embedding], [mem_embedding])[0][0]
                similarities.append((memory, similarity))
            except (json.JSONDecodeError, Exception):
                continue
        
        # Sort by similarity and get top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_memories = similarities[:top_k]
        
        # Also get user profile
        profile = {}
        profile_entries = self.db.query(UserProfile).all()
        for entry in profile_entries:
            try:
                profile[entry.key] = json.loads(entry.value) if entry.value.startswith('[') else entry.value
            except json.JSONDecodeError:
                profile[entry.key] = entry.value
        
        return {
            "memories": [
                {
                    "content": mem.content,
                    "category": mem.category,
                    "relevance": float(sim),
                    "created_at": mem.created_at.isoformat()
                }
                for mem, sim in top_memories
            ],
            "user_profile": profile
        }