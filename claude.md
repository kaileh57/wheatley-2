# Wheatley 2.0 - Personal AI Assistant System Design

## Executive Summary

Wheatley 2.0 is a personal AI assistant designed for single-user deployment. By focusing on one user, the system can be deeply personalized, more responsive, and simpler to maintain. Built around a Raspberry Pi hub with Gemini 2.5 Flash and MCP integration, it provides an always-on, privacy-focused assistant that learns and adapts to your specific needs.

## Simplified Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Personal Interfaces                          │
├──────────────────────────┬──────────────────────────────────────────┤
│   Hardware Device        │         Web Portal                       │
│  (Raspberry Pi Hub)      │    (Lightweight SPA)                     │
│  - Always Listening      │    - Quick Access                       │
│  - Ambient Display       │    - Mobile Friendly                    │
│  - Physical Controls     │    - Same Experience                    │
└───────────┬──────────────┴──────────────────────┬───────────────────┘
            │                                      │
            │        WebSocket (Local Network)    │
            │           Simple API Key            │
            ▼                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Personal Assistant Core (Raspberry Pi)            │
├─────────────────────────────────────────────────────────────────────┤
│  FastAPI Application                                                 │
│  ├── Voice Pipeline (Always On)                                     │
│  ├── Personal Agent (Your Dedicated AI)                             │
│  ├── Context Manager (Knows You)                                    │
│  ├── Task Executor (Your Background Worker)                         │
│  └── Memory System (Your Personal Knowledge Base)                   │
├─────────────────────────────────────────────────────────────────────┤
│  Local Services                                                      │
│  ├── SQLite (Simple, Fast, Local)                                   │
│  ├── Redis (Optional - for performance)                             │
│  ├── ChromaDB (Your Memory Embeddings)                              │
│  └── MCP Tools (Your Personal Toolkit)                              │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Simplifications & Improvements

### 1. Authentication → Simple API Key

```python
# No complex JWT, just a simple API key you set
API_KEY = os.getenv("WHEATLEY_API_KEY", "your-secret-key-here")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, key: str):
    if key != API_KEY:
        await websocket.close(code=1008)
        return
    
    await websocket.accept()
    # You're in - no user management needed
```

### 2. Personalized Wake Word & Voice

```python
class PersonalVoiceInterface:
    def __init__(self):
        # Train on YOUR voice for better accuracy
        self.wake_word = OpenWakeWord(
            model_path="models/hey_wheatley_personal.onnx",
            personalized=True
        )
        
        # Your preferred TTS voice settings
        self.tts = PersonalTTS(
            voice_id="your_preferred_voice",
            speed=1.1,  # You like it slightly faster
            pitch=0.9   # Slightly lower pitch
        )
        
    async def respond(self, text: str):
        # Add personal touches you prefer
        if self.is_morning():
            text = f"Good morning! {text}"
        
        await self.tts.speak(text)
```

### 3. Unified Personal Memory

```python
class PersonalMemory:
    def __init__(self):
        # Single user = simpler database schema
        self.db = sqlite3.connect("wheatley_personal.db")
        self.vector_store = ChromaDB(collection="my_memories")
        
    async def remember(self, content: str, context: dict):
        # No user isolation needed
        memory = {
            "content": content,
            "timestamp": datetime.now(),
            "location": context.get("location"),
            "activity": context.get("activity"),
            "mood": context.get("mood"),
            "importance": self.calculate_importance(content)
        }
        
        # Store everything - it's all yours
        await self.db.execute(
            "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?)",
            tuple(memory.values())
        )
        
        # Embed for semantic search
        embedding = await self.embed(content)
        await self.vector_store.add(embedding, memory)
    
    async def recall(self, query: str, limit: int = 10):
        # Search YOUR memories without filtering by user
        return await self.vector_store.search(query, limit)
```

### 4. Proactive Personal Assistant

```python
class ProactiveAssistant:
    def __init__(self):
        self.routines = PersonalRoutines()
        self.preferences = PersonalPreferences()
        
    async def ambient_awareness(self):
        """Always running, learning your patterns"""
        while True:
            current_context = await self.get_context()
            
            # Morning routine
            if current_context.is_morning and not self.morning_briefing_done:
                await self.morning_briefing()
            
            # Proactive reminders based on YOUR patterns
            if current_context.location == "kitchen" and current_context.time_since_last_meal > 6:
                await self.suggest("It's been a while since you ate. Want me to suggest something?")
            
            # Learn from your behavior
            await self.learn_pattern(current_context)
            
            await asyncio.sleep(60)  # Check every minute
    
    async def morning_briefing(self):
        """Your personalized morning update"""
        briefing = await self.generate_briefing(
            include_weather=True,
            include_calendar=True,
            include_news_topics=self.preferences.news_interests,
            include_reminders=True
        )
        
        await self.speak(briefing)
        self.morning_briefing_done = True
```

### 5. Simplified Task System

```python
class PersonalTaskExecutor:
    def __init__(self):
        # No multi-user queue management needed
        self.active_tasks = {}
        
    async def execute(self, task_description: str):
        task_id = str(uuid.uuid4())
        
        # Immediate feedback
        await self.notify("I'll work on that")
        
        # Execute in background
        asyncio.create_task(
            self._run_task(task_id, task_description)
        )
        
        return task_id
    
    async def _run_task(self, task_id: str, description: str):
        # Use all available resources - you're the only user
        result = await self.gemini.generate(
            model="gemini-2.5-flash",
            prompt=description,
            thinking_budget=1.0,  # Max thinking for your tasks
            tools=self.all_tools  # All tools available
        )
        
        # Direct notification - no user routing needed
        await self.notify_completion(result)
```

### 6. Personal Context Engine

```python
class PersonalContext:
    def __init__(self):
        self.current_state = {
            "location": "home",
            "activity": "working",
            "last_interaction": None,
            "mood": "neutral",
            "energy_level": "normal"
        }
        
        # Your personal patterns
        self.patterns = {
            "work_hours": (9, 17),
            "sleep_schedule": (23, 7),
            "meal_times": [8, 12, 18],
            "exercise_time": 7,
            "focus_periods": [(10, 12), (14, 16)]
        }
    
    async def update(self, signals: dict):
        """Update context from various signals"""
        # Time-based updates
        current_hour = datetime.now().hour
        
        if self.patterns["work_hours"][0] <= current_hour < self.patterns["work_hours"][1]:
            self.current_state["activity"] = "working"
        
        # Voice-based mood detection
        if signals.get("voice_energy"):
            self.current_state["mood"] = self.analyze_mood(signals["voice_energy"])
        
        # Location from network or GPS
        if signals.get("wifi_network") == "HOME_NETWORK":
            self.current_state["location"] = "home"
```

### 7. Optimized Hardware Setup

```python
# Since it's just for you, we can optimize hardware usage
class HardwareOptimizations:
    def __init__(self):
        # Use all available cores for your requests
        self.cpu_cores = multiprocessing.cpu_count()
        
        # Aggressive caching since memory isn't shared
        self.cache_size = "2GB"  
        
        # Pre-load your common models
        self.preloaded_models = {
            "whisper": load_model("small.en"),
            "embeddings": load_model("all-MiniLM-L6-v2"),
            "sentiment": load_model("sentiment-analysis")
        }
        
        # Your personal LED patterns
        self.led_patterns = {
            "listening": "breathing_blue",
            "thinking": "spinning_white",
            "speaking": "pulsing_green"
        }
```

### 8. Simplified Deployment

```yaml
# docker-compose.yml - Much simpler!
version: '3.8'

services:
  wheatley:
    build: .
    ports:
      - "8000:8000"  # API
      - "3000:3000"  # Web UI
    environment:
      - WHEATLEY_API_KEY=${WHEATLEY_API_KEY}
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    volumes:
      - ./data:/app/data  # Your personal data
      - ./memories:/app/memories  # Your memories
      - ./config.yaml:/app/config.yaml  # Your preferences
    devices:
      - /dev/snd:/dev/snd  # Audio access
    restart: unless-stopped

  # That's it! No multiple services needed
```

### 9. Personal Configuration

```yaml
# config.yaml - Your personal preferences
assistant:
  name: "Wheatley"
  personality: "friendly, concise, proactive"
  
voice:
  wake_word: "hey wheatley"
  tts_voice: "nova"
  tts_speed: 1.1
  language: "en-US"
  
preferences:
  morning_briefing_time: "07:00"
  news_sources: ["hackernews", "techcrunch"]
  weather_location: "Seattle"
  units: "imperial"
  
  # Your personal interests for context
  interests:
    - programming
    - AI/ML
    - robotics
    - home automation
  
  # Communication style
  communication:
    formality: "casual"
    humor: "enabled"
    verbosity: "concise"
  
routines:
  morning:
    - weather
    - calendar
    - news_summary
    - coffee_reminder
  
  evening:
    - tomorrow_prep
    - relaxation_suggestion
  
integrations:
  calendar: "google"
  email: "gmail"
  home_automation: "home_assistant"
  music: "spotify"
```

### 10. Privacy-First Local Processing

```python
class LocalFirstProcessing:
    def __init__(self):
        # Everything runs locally when possible
        self.local_models = {
            "intent": load_local_model("intent_classification"),
            "ner": load_local_model("named_entity_recognition"),
            "sentiment": load_local_model("sentiment_analysis")
        }
        
    async def process_request(self, text: str):
        # Try local processing first
        intent = self.local_models["intent"].predict(text)
        
        if intent.confidence > 0.9 and intent.type in LOCAL_INTENTS:
            # Handle locally without external API calls
            return await self.handle_local(intent, text)
        else:
            # Use Gemini for complex reasoning
            return await self.gemini_process(text)
```

## Unique Personal Features

### 1. Ambient Mode
- Always listening (with privacy)
- Learns your daily patterns
- Proactive suggestions based on context
- No "activation" needed for certain contexts

### 2. Personal Knowledge Graph
```python
# Build a graph of YOUR world
personal_graph = {
    "people": {
        "John": {"relationship": "brother", "birthday": "March 15"},
        "Sarah": {"relationship": "colleague", "project": "AI Research"}
    },
    "places": {
        "office": {"address": "...", "commute_time": 25},
        "gym": {"schedule": "MWF 7am"}
    },
    "projects": {
        "wheatley": {"status": "active", "priority": "high"}
    }
}
```

### 3. Continuous Learning
- Every interaction improves the system
- Learns your speech patterns
- Adapts to your preferences
- No privacy concerns - it's all your data

### 4. Hardware Integration
```python
# Physical buttons for common actions
GPIO_BUTTONS = {
    17: "pause_resume",  # Pause/Resume current task
    27: "privacy_mode",  # Temporary disable listening
    22: "quick_note",    # Record a quick note
    23: "timer_stop"     # Stop any active timer
}

# Ambient display shows relevant info
DISPLAY_MODES = {
    "default": show_time_and_weather,
    "working": show_focus_timer,
    "cooking": show_active_timers,
    "morning": show_briefing
}
```

## Simplified Implementation Timeline

### Week 1-2: Core Personal System
- Basic FastAPI server with simple auth
- SQLite database setup
- Wake word detection
- Basic STT/TTS pipeline

### Week 3-4: Personal AI Integration
- Gemini 2.5 Flash integration
- Personal memory system
- Context awareness
- MCP tools setup

### Week 5-6: Personal Features
- Proactive routines
- Pattern learning
- Personal knowledge graph
- Hardware integration

### Week 7-8: Polish & Daily Use
- Web interface
- Mobile access
- Backup system
- Personal optimizations

## Benefits of Single-User Design

1. **Performance**: All resources dedicated to you
2. **Privacy**: Your data never leaves your network
3. **Personalization**: Deep learning of your patterns
4. **Simplicity**: No user management complexity
5. **Reliability**: Fewer moving parts
6. **Cost**: Optimized for single user = lower costs
7. **Latency**: No multi-user overhead
8. **Features**: Can add very personal features

## Personal Data Backup

```python
# Simple backup since it's just your data
class PersonalBackup:
    async def daily_backup(self):
        backup_path = f"/backups/wheatley_{date.today()}.tar.gz"
        
        # Backup your personal data
        await compress_directory("/app/data", backup_path)
        
        # Optional: sync to your personal cloud
        if self.cloud_backup_enabled:
            await sync_to_cloud(backup_path)
```

This personal-first design makes Wheatley 2.0 a true digital companion that knows you, adapts to you, and proactively helps you throughout your day - all while keeping things simple and private.