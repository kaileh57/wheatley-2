"""
Configuration settings for Wheatley 2.0
"""
import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings"""
    
    # API Keys
    wheatley_api_key: str = Field(default="your-secret-key-here", env="WHEATLEY_API_KEY")
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Gemini settings
    gemini_model: str = "gemini-2.5-flash"
    gemini_thinking_budget: float = 1.0  # Max thinking budget for personal use
    gemini_temperature: float = 0.7
    gemini_max_tokens: int = 2048
    
    # TTS settings (Gemini Native Audio)
    tts_model: str = "gemini-2.5-flash-preview-tts"
    tts_voice: str = "nova"  # Default voice
    tts_speed: float = 1.1
    
    # STT settings (Whisper)
    whisper_model: str = "small.en"
    whisper_device: str = "cpu"  # or "cuda" if GPU available
    
    # Wake word settings
    wake_word: str = "hey wheatley"
    wake_word_threshold: float = 0.5
    
    # Memory settings
    sqlite_db_path: str = "data/wheatley_personal.db"
    chroma_db_path: str = "data/chroma_db"
    max_memory_items: int = 10000
    
    # Hardware settings
    led_enabled: bool = False  # Set to True on Raspberry Pi
    button_pins: List[int] = [17, 27, 22, 23]  # GPIO pins for buttons
    
    # Context settings
    morning_briefing_time: str = "07:00"
    location_home_wifi: str = "HOME_NETWORK"
    
    # Redis settings (optional)
    redis_url: Optional[str] = None
    cache_ttl: int = 3600  # 1 hour
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/wheatley.log"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Create global settings instance
settings = Settings()

# User preferences configuration (loaded from YAML)
import yaml
from pathlib import Path

def load_user_config():
    """Load user preferences from config.yaml"""
    config_path = Path("config.yaml")
    if not config_path.exists():
        # Create default config
        default_config = {
            "assistant": {
                "name": "Wheatley",
                "personality": "friendly, concise, proactive"
            },
            "voice": {
                "wake_word": "hey wheatley",
                "tts_voice": "nova",
                "tts_speed": 1.1,
                "language": "en-US"
            },
            "preferences": {
                "morning_briefing_time": "07:00",
                "news_sources": ["hackernews", "techcrunch"],
                "weather_location": "Seattle",
                "units": "imperial",
                "interests": [
                    "programming",
                    "AI/ML",
                    "robotics",
                    "home automation"
                ],
                "communication": {
                    "formality": "casual",
                    "humor": "enabled",
                    "verbosity": "concise"
                }
            },
            "routines": {
                "morning": [
                    "weather",
                    "calendar",
                    "news_summary",
                    "coffee_reminder"
                ],
                "evening": [
                    "tomorrow_prep",
                    "relaxation_suggestion"
                ]
            },
            "integrations": {
                "calendar": "google",
                "email": "gmail",
                "home_automation": "home_assistant",
                "music": "spotify"
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        return default_config
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Load user config on startup
user_config = load_user_config() 