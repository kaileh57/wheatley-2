"""
Personal Memory System using SQLite and ChromaDB
"""
import asyncio
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

from backend.config import settings
from backend.models.schemas import Memory

logger = logging.getLogger(__name__)

Base = declarative_base()


class MemoryDB(Base):
    """SQLite memory model"""
    __tablename__ = 'memories'
    
    id = Column(Integer, primary_key=True)
    content = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.now)
    context = Column(JSON)
    importance = Column(Float, default=0.5)
    category = Column(String)
    tags = Column(JSON)


class MemoryService:
    """Personal memory management service"""
    
    def __init__(self):
        # Ensure data directory exists
        data_dir = Path(settings.sqlite_db_path).parent
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite
        self.engine = create_async_engine(
            f"sqlite+aiosqlite:///{settings.sqlite_db_path}",
            echo=False
        )
        self.SessionLocal = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=settings.chroma_db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="personal_memories",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize embedding model
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create tables
        asyncio.create_task(self._create_tables())
        
        logger.info("Memory service initialized")
    
    async def _create_tables(self):
        """Create database tables"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def store_memory(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Memory:
        """Store a new memory"""
        # Create memory object
        memory = Memory(
            content=content,
            context=context,
            importance=importance,
            category=category,
            tags=tags or []
        )
        
        # Generate embedding
        embedding = self.embedder.encode(content).tolist()
        memory.embedding = embedding
        
        # Store in SQLite
        async with self.SessionLocal() as session:
            db_memory = MemoryDB(
                content=content,
                timestamp=memory.timestamp,
                context=context,
                importance=importance,
                category=category,
                tags=tags
            )
            session.add(db_memory)
            await session.commit()
            await session.refresh(db_memory)
            memory.id = db_memory.id
        
        # Store in ChromaDB
        self.collection.add(
            ids=[str(memory.id)],
            embeddings=[embedding],
            documents=[content],
            metadatas=[{
                "timestamp": memory.timestamp.isoformat(),
                "importance": importance,
                "category": category or "general",
                "tags": json.dumps(tags or [])
            }]
        )
        
        logger.info(f"Stored memory {memory.id}: {content[:50]}...")
        
        # Manage memory size
        await self._manage_memory_size()
        
        return memory
    
    async def search_memories(
        self,
        query: str,
        limit: int = 10,
        importance_threshold: float = 0.0
    ) -> List[Memory]:
        """Search memories using semantic similarity"""
        # Generate query embedding
        query_embedding = self.embedder.encode(query).tolist()
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where={"importance": {"$gte": importance_threshold}}
        )
        
        if not results['ids'][0]:
            return []
        
        # Fetch full memory data from SQLite
        memory_ids = [int(id_) for id_ in results['ids'][0]]
        
        async with self.SessionLocal() as session:
            memories = []
            for idx, memory_id in enumerate(memory_ids):
                result = await session.execute(
                    f"SELECT * FROM memories WHERE id = {memory_id}"
                )
                row = result.first()
                if row:
                    memory = Memory(
                        id=row.id,
                        content=row.content,
                        timestamp=row.timestamp,
                        context=row.context,
                        importance=row.importance,
                        category=row.category,
                        tags=row.tags or [],
                        embedding=results['embeddings'][0][idx] if results['embeddings'] else None
                    )
                    memories.append(memory)
        
        return memories
    
    async def get_memories_by_category(
        self,
        category: str,
        limit: int = 20
    ) -> List[Memory]:
        """Get memories by category"""
        async with self.SessionLocal() as session:
            result = await session.execute(
                f"SELECT * FROM memories WHERE category = '{category}' "
                f"ORDER BY timestamp DESC LIMIT {limit}"
            )
            
            memories = []
            for row in result:
                memory = Memory(
                    id=row.id,
                    content=row.content,
                    timestamp=row.timestamp,
                    context=row.context,
                    importance=row.importance,
                    category=row.category,
                    tags=row.tags or []
                )
                memories.append(memory)
        
        return memories
    
    async def get_recent_memories(
        self,
        hours: int = 24,
        limit: int = 50
    ) -> List[Memory]:
        """Get recent memories within specified hours"""
        async with self.SessionLocal() as session:
            cutoff_time = datetime.now().timestamp() - (hours * 3600)
            result = await session.execute(
                f"SELECT * FROM memories WHERE "
                f"CAST(strftime('%s', timestamp) AS INTEGER) > {cutoff_time} "
                f"ORDER BY timestamp DESC LIMIT {limit}"
            )
            
            memories = []
            for row in result:
                memory = Memory(
                    id=row.id,
                    content=row.content,
                    timestamp=row.timestamp,
                    context=row.context,
                    importance=row.importance,
                    category=row.category,
                    tags=row.tags or []
                )
                memories.append(memory)
        
        return memories
    
    async def update_memory_importance(
        self,
        memory_id: int,
        new_importance: float
    ):
        """Update the importance of a memory"""
        async with self.SessionLocal() as session:
            await session.execute(
                f"UPDATE memories SET importance = {new_importance} "
                f"WHERE id = {memory_id}"
            )
            await session.commit()
        
        # Update in ChromaDB
        self.collection.update(
            ids=[str(memory_id)],
            metadatas=[{"importance": new_importance}]
        )
    
    async def delete_memory(self, memory_id: int):
        """Delete a memory"""
        async with self.SessionLocal() as session:
            await session.execute(
                f"DELETE FROM memories WHERE id = {memory_id}"
            )
            await session.commit()
        
        # Delete from ChromaDB
        self.collection.delete(ids=[str(memory_id)])
    
    async def _manage_memory_size(self):
        """Manage memory size by removing old, low-importance memories"""
        async with self.SessionLocal() as session:
            # Count total memories
            result = await session.execute("SELECT COUNT(*) FROM memories")
            count = result.scalar()
            
            if count > settings.max_memory_items:
                # Delete oldest, least important memories
                excess = count - settings.max_memory_items
                result = await session.execute(
                    f"SELECT id FROM memories "
                    f"ORDER BY importance ASC, timestamp ASC "
                    f"LIMIT {excess}"
                )
                
                ids_to_delete = [row[0] for row in result]
                
                for memory_id in ids_to_delete:
                    await self.delete_memory(memory_id)
                
                logger.info(f"Cleaned up {excess} old memories")
    
    async def create_memory_summary(
        self,
        time_period_hours: int = 24
    ) -> str:
        """Create a summary of recent memories"""
        recent_memories = await self.get_recent_memories(time_period_hours)
        
        if not recent_memories:
            return "No significant memories in the specified period."
        
        # Group by category
        categories = {}
        for memory in recent_memories:
            category = memory.category or "general"
            if category not in categories:
                categories[category] = []
            categories[category].append(memory.content)
        
        # Create summary
        summary_parts = []
        for category, contents in categories.items():
            summary_parts.append(f"{category.title()}:")
            for content in contents[:3]:  # Top 3 per category
                summary_parts.append(f"  - {content[:100]}...")
        
        return "\n".join(summary_parts)
    
    async def build_personal_knowledge_graph(self) -> Dict[str, Any]:
        """Build a knowledge graph from memories"""
        # This is a simplified version - could be enhanced with NER and relationship extraction
        async with self.SessionLocal() as session:
            result = await session.execute(
                "SELECT content, tags, category FROM memories "
                "WHERE importance > 0.5"
            )
            
            entities = {}
            relationships = []
            
            for row in result:
                content = row.content
                tags = row.tags or []
                category = row.category or "general"
                
                # Extract entities (simplified)
                words = content.lower().split()
                for word in words:
                    if word in ["i", "me", "my"]:
                        continue
                    if len(word) > 4 and word.isalpha():
                        if word not in entities:
                            entities[word] = {
                                "count": 0,
                                "categories": set(),
                                "related_tags": set()
                            }
                        entities[word]["count"] += 1
                        entities[word]["categories"].add(category)
                        entities[word]["related_tags"].update(tags)
            
            # Convert sets to lists for JSON serialization
            for entity in entities.values():
                entity["categories"] = list(entity["categories"])
                entity["related_tags"] = list(entity["related_tags"])
            
            return {
                "entities": entities,
                "relationships": relationships
            } 