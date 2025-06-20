"""
Personal Context Manager for tracking user state and patterns
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging
import json
from pathlib import Path

from backend.config import settings, user_config
from backend.models.schemas import UserContext, PersonalPattern

logger = logging.getLogger(__name__)


class ContextManager:
    """Manages user context and behavioral patterns"""
    
    def __init__(self):
        self.current_context = UserContext()
        self.patterns: Dict[str, PersonalPattern] = {}
        self.state_file = Path("data/context_state.json")
        self.patterns_file = Path("data/personal_patterns.json")
        
        # Ensure data directory exists
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load saved state
        self._load_state()
        
        # Define default patterns
        self._initialize_default_patterns()
        
        # Start background context updater
        asyncio.create_task(self._context_update_loop())
        
        logger.info("Context manager initialized")
    
    def _initialize_default_patterns(self):
        """Initialize default personal patterns from config"""
        # Work hours pattern
        self.patterns["work_hours"] = PersonalPattern(
            pattern_type="work_hours",
            time_ranges=[[9, 17]],  # 9 AM to 5 PM
            confidence=0.5
        )
        
        # Sleep schedule
        self.patterns["sleep_schedule"] = PersonalPattern(
            pattern_type="sleep_schedule",
            time_ranges=[[23, 24], [0, 7]],  # 11 PM to 7 AM
            confidence=0.5
        )
        
        # Meal times
        self.patterns["meal_times"] = PersonalPattern(
            pattern_type="meal_times",
            time_ranges=[[8, 9], [12, 13], [18, 19]],  # Breakfast, lunch, dinner
            confidence=0.5
        )
        
        # Exercise time
        self.patterns["exercise_time"] = PersonalPattern(
            pattern_type="exercise_time",
            time_ranges=[[7, 8]],  # 7-8 AM
            confidence=0.3
        )
        
        # Focus periods
        self.patterns["focus_periods"] = PersonalPattern(
            pattern_type="focus_periods",
            time_ranges=[[10, 12], [14, 16]],  # Morning and afternoon focus
            confidence=0.4
        )
    
    def _load_state(self):
        """Load saved context state and patterns"""
        # Load context state
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state_data = json.load(f)
                    self.current_context = UserContext(**state_data)
                logger.info("Loaded saved context state")
            except Exception as e:
                logger.error(f"Error loading context state: {e}")
        
        # Load patterns
        if self.patterns_file.exists():
            try:
                with open(self.patterns_file, 'r') as f:
                    patterns_data = json.load(f)
                    for pattern_type, pattern_data in patterns_data.items():
                        self.patterns[pattern_type] = PersonalPattern(**pattern_data)
                logger.info("Loaded saved patterns")
            except Exception as e:
                logger.error(f"Error loading patterns: {e}")
    
    def _save_state(self):
        """Save current context state and patterns"""
        # Save context
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.current_context.dict(), f, default=str)
        except Exception as e:
            logger.error(f"Error saving context state: {e}")
        
        # Save patterns
        try:
            patterns_data = {
                name: pattern.dict() for name, pattern in self.patterns.items()
            }
            with open(self.patterns_file, 'w') as f:
                json.dump(patterns_data, f)
        except Exception as e:
            logger.error(f"Error saving patterns: {e}")
    
    async def _context_update_loop(self):
        """Background task to update context periodically"""
        while True:
            try:
                await self._update_context_from_time()
                await self._detect_activity_patterns()
                self._save_state()
                await asyncio.sleep(60)  # Update every minute
            except Exception as e:
                logger.error(f"Error in context update loop: {e}")
                await asyncio.sleep(60)
    
    async def _update_context_from_time(self):
        """Update context based on current time"""
        current_time = datetime.now()
        current_hour = current_time.hour
        
        # Update time-based activity
        old_activity = self.current_context.activity
        
        # Check work hours
        if self._is_in_time_range(current_hour, self.patterns["work_hours"].time_ranges):
            self.current_context.activity = "working"
        # Check sleep time
        elif self._is_in_time_range(current_hour, self.patterns["sleep_schedule"].time_ranges):
            self.current_context.activity = "sleeping"
        # Check meal times
        elif self._is_in_time_range(current_hour, self.patterns["meal_times"].time_ranges):
            self.current_context.activity = "eating"
        # Check exercise time
        elif self._is_in_time_range(current_hour, self.patterns["exercise_time"].time_ranges):
            self.current_context.activity = "exercising"
        else:
            self.current_context.activity = "idle"
        
        # Log activity changes
        if old_activity != self.current_context.activity:
            logger.info(f"Activity changed from {old_activity} to {self.current_context.activity}")
        
        # Update current time
        self.current_context.current_time = current_time
    
    def _is_in_time_range(self, hour: int, time_ranges: List[List[int]]) -> bool:
        """Check if current hour is within any of the time ranges"""
        for start, end in time_ranges:
            if start <= end:
                if start <= hour < end:
                    return True
            else:  # Handle ranges that cross midnight
                if hour >= start or hour < end:
                    return True
        return False
    
    async def _detect_activity_patterns(self):
        """Detect and update activity patterns based on user behavior"""
        # This is where we would analyze user behavior and update patterns
        # For now, we'll just slowly increase confidence for current activities
        current_hour = datetime.now().hour
        
        for pattern_name, pattern in self.patterns.items():
            if self._is_in_time_range(current_hour, pattern.time_ranges):
                # Increase confidence slightly
                pattern.confidence = min(1.0, pattern.confidence + 0.01)
                pattern.observations += 1
    
    async def update_from_signals(self, signals: Dict[str, Any]):
        """Update context from various signals"""
        # Update location from WiFi
        if "wifi_network" in signals:
            if signals["wifi_network"] == settings.location_home_wifi:
                self.current_context.location = "home"
            else:
                self.current_context.location = "away"
        
        # Update from voice energy (mood detection)
        if "voice_energy" in signals:
            energy = signals["voice_energy"]
            if energy > 0.7:
                self.current_context.mood = "energetic"
            elif energy < 0.3:
                self.current_context.mood = "tired"
            else:
                self.current_context.mood = "neutral"
        
        # Update energy level
        if "activity_level" in signals:
            self.current_context.energy_level = signals["activity_level"]
        
        # Update last interaction time
        if "interaction" in signals and signals["interaction"]:
            self.current_context.last_interaction = datetime.now()
        
        # Update environment data
        self.current_context.environment.update(signals)
        
        # Save updated state
        self._save_state()
    
    def get_current_context(self) -> UserContext:
        """Get the current user context"""
        return self.current_context
    
    def get_activity_probability(self, activity: str) -> float:
        """Get the probability of a specific activity at current time"""
        current_hour = datetime.now().hour
        
        # Map activities to patterns
        activity_pattern_map = {
            "working": "work_hours",
            "sleeping": "sleep_schedule",
            "eating": "meal_times",
            "exercising": "exercise_time",
            "focusing": "focus_periods"
        }
        
        pattern_name = activity_pattern_map.get(activity)
        if not pattern_name or pattern_name not in self.patterns:
            return 0.0
        
        pattern = self.patterns[pattern_name]
        if self._is_in_time_range(current_hour, pattern.time_ranges):
            return pattern.confidence
        return 0.0
    
    def should_interrupt(self) -> bool:
        """Determine if it's appropriate to interrupt the user"""
        # Don't interrupt during sleep
        if self.current_context.activity == "sleeping":
            return False
        
        # Don't interrupt during focus periods with high confidence
        if (self.current_context.activity == "working" and 
            self.get_activity_probability("focusing") > 0.7):
            return False
        
        # Check if recently interacted
        if self.current_context.last_interaction:
            time_since_interaction = datetime.now() - self.current_context.last_interaction
            if time_since_interaction < timedelta(minutes=5):
                return False
        
        return True
    
    def get_contextual_greeting(self) -> str:
        """Get a contextual greeting based on time and activity"""
        hour = datetime.now().hour
        
        if 5 <= hour < 12:
            greeting = "Good morning"
        elif 12 <= hour < 17:
            greeting = "Good afternoon"
        elif 17 <= hour < 22:
            greeting = "Good evening"
        else:
            greeting = "Hello"
        
        # Add activity-based context
        if self.current_context.activity == "working":
            greeting += "! Hope your work is going well."
        elif self.current_context.activity == "exercising":
            greeting += "! Great job staying active!"
        elif self.current_context.activity == "eating":
            greeting += "! Enjoy your meal."
        else:
            greeting += "!"
        
        return greeting
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of current context and patterns"""
        return {
            "current_state": self.current_context.dict(),
            "patterns": {
                name: {
                    "confidence": pattern.confidence,
                    "observations": pattern.observations,
                    "active": self._is_in_time_range(
                        datetime.now().hour,
                        pattern.time_ranges
                    )
                }
                for name, pattern in self.patterns.items()
            },
            "should_interrupt": self.should_interrupt(),
            "contextual_greeting": self.get_contextual_greeting()
        } 