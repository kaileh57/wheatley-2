"""
Asynchronous Task Executor for background processing
"""
import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List
import logging
from concurrent.futures import ThreadPoolExecutor

from backend.models.schemas import Task, TaskStatus
from backend.services.agent import PersonalAgent

logger = logging.getLogger(__name__)


class TaskExecutor:
    """Manages and executes background tasks"""
    
    def __init__(self, agent: PersonalAgent):
        self.agent = agent
        self.active_tasks: Dict[str, Task] = {}
        self.task_queue = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.notification_callbacks: List[Callable] = []
        
        # Start task processor
        asyncio.create_task(self._task_processor())
        
        logger.info("Task executor initialized")
    
    async def _task_processor(self):
        """Background task processor"""
        while True:
            try:
                task_id = await self.task_queue.get()
                if task_id in self.active_tasks:
                    task = self.active_tasks[task_id]
                    await self._execute_task(task)
            except Exception as e:
                logger.error(f"Error in task processor: {e}")
            finally:
                self.task_queue.task_done()
    
    async def create_task(
        self,
        description: str,
        task_type: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Task:
        """Create and queue a new task"""
        task_id = str(uuid.uuid4())
        
        task = Task(
            id=task_id,
            description=description,
            status=TaskStatus.PENDING,
            metadata=metadata or {}
        )
        
        self.active_tasks[task_id] = task
        await self.task_queue.put(task_id)
        
        logger.info(f"Created task {task_id}: {description}")
        
        # Send immediate acknowledgment
        await self._notify_task_update(task, "I'll work on that")
        
        return task
    
    async def _execute_task(self, task: Task):
        """Execute a specific task"""
        try:
            # Update task status
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            await self._notify_task_update(task, "Working on your request...")
            
            # Determine task type and execute
            result = await self._process_task_by_type(task)
            
            # Update task completion
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            task.progress = 1.0
            
            # Notify completion with sound
            await self._notify_task_completion(task)
            
        except Exception as e:
            logger.error(f"Error executing task {task.id}: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            
            await self._notify_task_update(
                task,
                f"I encountered an error with that task: {str(e)}"
            )
    
    async def _process_task_by_type(self, task: Task) -> Any:
        """Process task based on its type"""
        description = task.description.lower()
        
        # Research tasks
        if any(word in description for word in ["research", "find out", "learn about", "investigate"]):
            return await self._process_research_task(task)
        
        # Report/summary tasks
        elif any(word in description for word in ["report", "summary", "summarize", "compile"]):
            return await self._process_report_task(task)
        
        # Reminder tasks
        elif any(word in description for word in ["remind", "alert", "notify"]):
            return await self._process_reminder_task(task)
        
        # Analysis tasks
        elif any(word in description for word in ["analyze", "compare", "evaluate"]):
            return await self._process_analysis_task(task)
        
        # Default: use agent to process
        else:
            return await self._process_general_task(task)
    
    async def _process_research_task(self, task: Task) -> Dict[str, Any]:
        """Process a research task"""
        # Extract research topic
        topic = task.description.replace("research", "").replace("find out about", "").strip()
        
        # Update progress
        task.progress = 0.2
        await self._notify_task_update(task, f"Researching {topic}...")
        
        # Perform web search
        search_results = await self.agent.search_web(topic)
        task.progress = 0.6
        
        # Generate comprehensive report
        report_prompt = f"""Based on the following research about {topic}:
        
{search_results}

Please create a comprehensive summary including:
1. Key findings
2. Important facts
3. Recent developments
4. Relevant implications
5. Sources for further reading"""

        from backend.services.context import ContextManager
        context_manager = ContextManager()
        context = context_manager.get_current_context()
        
        response, _, _ = await self.agent.process_request(
            report_prompt,
            context,
            include_memory=False
        )
        
        task.progress = 1.0
        
        return {
            "topic": topic,
            "search_results": search_results,
            "summary": response,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _process_report_task(self, task: Task) -> Dict[str, Any]:
        """Process a report generation task"""
        # This would generate various types of reports
        report_type = "general"
        
        if "daily" in task.description:
            report_type = "daily_summary"
        elif "weekly" in task.description:
            report_type = "weekly_summary"
        
        # Generate report based on memories and context
        from backend.services.memory import MemoryService
        memory_service = MemoryService()
        
        # Get relevant memories
        time_period = 24 if report_type == "daily_summary" else 168
        memory_summary = await memory_service.create_memory_summary(time_period)
        
        report = {
            "type": report_type,
            "period": f"Last {time_period} hours",
            "summary": memory_summary,
            "generated_at": datetime.now().isoformat()
        }
        
        return report
    
    async def _process_reminder_task(self, task: Task) -> Dict[str, Any]:
        """Process a reminder task"""
        # Parse reminder details from description
        # This is a simplified version - could use NLP for better parsing
        import re
        
        time_match = re.search(r'at (\d{1,2}:\d{2})', task.description)
        if time_match:
            reminder_time = time_match.group(1)
        else:
            reminder_time = "soon"
        
        message = task.description.replace(time_match.group(0) if time_match else "", "").strip()
        
        # Schedule the reminder (simplified - would use APScheduler in production)
        reminder_data = {
            "message": message,
            "scheduled_time": reminder_time,
            "created_at": datetime.now().isoformat()
        }
        
        return reminder_data
    
    async def _process_analysis_task(self, task: Task) -> Dict[str, Any]:
        """Process an analysis task"""
        # Extract what needs to be analyzed
        subject = task.description.replace("analyze", "").replace("compare", "").strip()
        
        # Perform analysis using agent
        analysis_prompt = f"""Please perform a detailed analysis of: {subject}
        
Include:
1. Key components or aspects
2. Strengths and weaknesses
3. Patterns or trends
4. Recommendations or insights
5. Conclusion"""

        from backend.services.context import ContextManager
        context_manager = ContextManager()
        context = context_manager.get_current_context()
        
        response, _, _ = await self.agent.process_request(
            analysis_prompt,
            context,
            include_memory=True
        )
        
        return {
            "subject": subject,
            "analysis": response,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _process_general_task(self, task: Task) -> Any:
        """Process a general task using the agent"""
        from backend.services.context import ContextManager
        context_manager = ContextManager()
        context = context_manager.get_current_context()
        
        response, tool_results, _ = await self.agent.process_request(
            task.description,
            context,
            include_memory=True
        )
        
        return {
            "response": response,
            "tool_results": tool_results,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _notify_task_update(self, task: Task, message: str):
        """Send task update notification"""
        update = {
            "task_id": task.id,
            "status": task.status.value,
            "progress": task.progress,
            "message": message
        }
        
        for callback in self.notification_callbacks:
            try:
                await callback("task_update", update)
            except Exception as e:
                logger.error(f"Error in notification callback: {e}")
    
    async def _notify_task_completion(self, task: Task):
        """Send task completion notification with sound"""
        message = f"Task completed: {task.description[:50]}..."
        
        completion_data = {
            "task_id": task.id,
            "status": "completed",
            "message": message,
            "play_sound": True,
            "result_preview": str(task.result)[:200] if task.result else None
        }
        
        for callback in self.notification_callbacks:
            try:
                await callback("task_complete", completion_data)
            except Exception as e:
                logger.error(f"Error in completion callback: {e}")
    
    def register_notification_callback(self, callback: Callable):
        """Register a callback for task notifications"""
        self.notification_callbacks.append(callback)
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a specific task by ID"""
        return self.active_tasks.get(task_id)
    
    def get_active_tasks(self) -> List[Task]:
        """Get all active tasks"""
        return [
            task for task in self.active_tasks.values()
            if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]
        ]
    
    def get_completed_tasks(self, limit: int = 10) -> List[Task]:
        """Get recently completed tasks"""
        completed = [
            task for task in self.active_tasks.values()
            if task.status == TaskStatus.COMPLETED
        ]
        
        # Sort by completion time
        completed.sort(key=lambda t: t.completed_at or datetime.min, reverse=True)
        
        return completed[:limit]
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.now()
                
                await self._notify_task_update(
                    task,
                    "Task has been cancelled"
                )
                return True
        return False
    
    def cleanup_old_tasks(self, hours: int = 24):
        """Clean up old completed tasks"""
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        
        tasks_to_remove = []
        for task_id, task in self.active_tasks.items():
            if (task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] and
                task.completed_at and
                task.completed_at.timestamp() < cutoff_time):
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self.active_tasks[task_id]
        
        if tasks_to_remove:
            logger.info(f"Cleaned up {len(tasks_to_remove)} old tasks") 