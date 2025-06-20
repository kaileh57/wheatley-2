"""
Simple API key authentication for single-user system
"""
import logging
from typing import Optional

from fastapi import HTTPException, Security, WebSocket
from fastapi.security import APIKeyHeader, APIKeyQuery

from backend.config import settings

logger = logging.getLogger(__name__)

# API Key authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)


async def get_api_key(
    api_key_header: Optional[str] = Security(api_key_header),
    api_key_query: Optional[str] = Security(api_key_query),
) -> str:
    """Validate API key from header or query parameter"""
    api_key = api_key_header or api_key_query
    
    if not api_key:
        raise HTTPException(
            status_code=403,
            detail="API Key required"
        )
    
    if api_key != settings.wheatley_api_key:
        raise HTTPException(
            status_code=403,
            detail="Invalid API Key"
        )
    
    return api_key


async def validate_websocket_key(websocket: WebSocket, api_key: Optional[str]) -> bool:
    """Validate API key for WebSocket connection"""
    if not api_key:
        await websocket.close(code=1008, reason="API Key required")
        return False
    
    if api_key != settings.wheatley_api_key:
        await websocket.close(code=1008, reason="Invalid API Key")
        return False
    
    return True


def is_valid_api_key(api_key: str) -> bool:
    """Check if API key is valid"""
    return api_key == settings.wheatley_api_key 