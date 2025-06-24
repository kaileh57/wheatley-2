from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Keys
    anthropic_api_key: str
    gemini_api_key: str
    perplexity_api_key: Optional[str] = None
    openai_api_key: str
    
    # Auth
    secret_key: str
    user_password_hash: str
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Paths
    sandbox_path: str = "./sandboxes"
    db_path: str = "./data/wheatley.db"
    memory_path: str = "./data/memory_embeddings"
    
    # Models
    default_model: str = "claude-4-opus"
    simple_query_model: str = "gemini-2.5-flash"
    
    class Config:
        env_file = ".env"

settings = Settings()