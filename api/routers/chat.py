from fastapi import APIRouter, Depends, HTTPException
from api.dependencies import verify_api_key
from api.services.nlp.chat_agent import get_chat_agent

router = APIRouter()

@router.post("/chat")
def chat(payload: dict, _key: str = Depends(verify_api_key)):
    try:
        agent = get_chat_agent()
    except Exception:
        raise HTTPException(503, "Chat agent not available")
    messages = payload.get("messages", [])
    if not messages:
        raise HTTPException(400, "No messages provided")
    return agent.chat(messages)
