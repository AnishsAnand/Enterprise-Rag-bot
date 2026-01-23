"""
WebUI Tasks API - Title generation and other task endpoints
Compatible with OpenWebUI's /api/task endpoints
"""

import os
import logging
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Request, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.services.ai_service import ai_service

log = logging.getLogger(__name__)

router = APIRouter(tags=["Tasks"])


# ==================== Pydantic Models ====================

class MessageModel(BaseModel):
    """Message model for title generation"""
    role: str
    content: str


class TitleGenerationRequest(BaseModel):
    """Request for title generation"""
    model: Optional[str] = None
    messages: List[MessageModel]
    stream: Optional[bool] = False


class TitleGenerationResponse(BaseModel):
    """Response for title generation"""
    title: str


# ==================== Config ====================

DEFAULT_TITLE_GENERATION_PROMPT = """### Task:
Generate a concise, 3-5 word title with an emoji summarizing the chat history.
### Guidelines:
- The title should clearly represent the main theme or subject of the conversation.
- Use emojis that enhance understanding of the topic, but avoid quotation marks or special formatting.
- Write the title in the chat's primary language; default to English if multilingual.
- Focus on the most recent topic if the conversation shifts.
### Output:
JSON format: { "title": "<generated title>" }
### Chat History:
"""


def _format_messages_for_title(messages: List[MessageModel], max_messages: int = 5) -> str:
    """Format messages for title generation prompt"""
    # Take most recent messages
    recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages
    
    formatted = []
    for msg in recent_messages:
        role = msg.role.upper()
        content = msg.content[:500]  # Truncate long messages
        formatted.append(f"{role}: {content}")
    
    return "\n".join(formatted)


async def _generate_title_with_ai(messages: List[MessageModel]) -> str:
    """Generate title using AI service"""
    try:
        # Format the prompt
        messages_text = _format_messages_for_title(messages)
        prompt = DEFAULT_TITLE_GENERATION_PROMPT + messages_text
        
        # Try to use AI service
        if ai_service and hasattr(ai_service, 'generate_completion'):
            response = await ai_service.generate_completion(
                prompt=prompt,
                max_tokens=50,
                temperature=0.7
            )
            
            # Parse the response - try to extract title
            if response:
                import json
                import re
                
                # Try JSON parsing first
                try:
                    data = json.loads(response)
                    if "title" in data:
                        return data["title"]
                except:
                    pass
                
                # Try regex for JSON-like pattern
                match = re.search(r'"title"\s*:\s*"([^"]+)"', response)
                if match:
                    return match.group(1)
                
                # Just use the response as-is if it's short enough
                response = response.strip().strip('"').strip("'")
                if len(response) <= 60:
                    return response
        
        # Fallback: extract from first user message
        return _extract_title_from_first_message(messages)
        
    except Exception as e:
        log.warning(f"AI title generation failed: {e}, falling back to extraction")
        return _extract_title_from_first_message(messages)


def _extract_title_from_first_message(messages: List[MessageModel], max_length: int = 50) -> str:
    """Extract title from first user message as fallback"""
    for msg in messages:
        if msg.role == "user" and msg.content.strip():
            content = msg.content.strip()
            # Take first line
            title = content.split("\n")[0].strip()
            # Truncate if too long
            if len(title) > max_length:
                title = title[:max_length - 3].rsplit(" ", 1)[0] + "..."
            return title
    
    return "New Chat"


# ==================== Endpoints ====================

@router.get("/api/task/config")
async def get_task_config():
    """Get task configuration"""
    return {
        "ENABLE_TITLE_GENERATION": os.getenv("ENABLE_TITLE_GENERATION", "true").lower() == "true",
        "TITLE_GENERATION_PROMPT_TEMPLATE": os.getenv("TITLE_GENERATION_PROMPT_TEMPLATE", ""),
        "ENABLE_TAGS_GENERATION": False,
        "ENABLE_SEARCH_QUERY_GENERATION": False,
    }


@router.post("/api/task/title/completions")
async def generate_title(
    request: Request,
    form_data: TitleGenerationRequest,
):
    """
    Generate a title for a chat based on its messages.
    This is OpenWebUI-compatible endpoint.
    """
    # Check if title generation is enabled
    if os.getenv("ENABLE_TITLE_GENERATION", "true").lower() != "true":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Title generation is disabled"
        )
    
    if not form_data.messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No messages provided"
        )
    
    # Generate title
    title = await _generate_title_with_ai(form_data.messages)
    
    # Return in OpenWebUI-compatible format
    if form_data.stream:
        # Streaming response (simplified)
        async def stream_title():
            import json
            yield f"data: {json.dumps({'choices': [{'delta': {'content': title}}]})}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            stream_title(),
            media_type="text/event-stream"
        )
    
    return {
        "choices": [
            {
                "message": {
                    "content": title
                }
            }
        ]
    }


@router.post("/api/v1/chats/{chat_id}/title")
async def update_chat_title(
    chat_id: str,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Generate and update the title for a specific chat.
    """
    from app.services.chat_service import ChatService
    
    # Get current user ID
    user_id = request.headers.get("X-User-Id", "default_user")
    
    chat_service = ChatService()
    chat = chat_service.get_chat_by_id_and_user_id(chat_id, user_id, db)
    
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found"
        )
    
    # Extract messages from chat history
    history = chat.chat.get("history", {})
    messages_dict = history.get("messages", {})
    
    messages = []
    for msg_id, msg in messages_dict.items():
        messages.append(MessageModel(
            role=msg.get("role", "user"),
            content=msg.get("content", "")
        ))
    
    if not messages:
        return {"title": chat.title}
    
    # Generate new title
    new_title = await _generate_title_with_ai(messages)
    
    # Update the chat with new title
    chat_data = chat.chat.copy()
    chat_data["title"] = new_title
    
    updated_chat = chat_service.update_chat_by_id(chat_id, chat_data, db)
    
    return {"title": new_title, "chat_id": chat_id}
