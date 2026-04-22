from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.models.schemas import ChatRequest, ChatResponse
from app.services.chat_service import chat_service

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest):
    result = await chat_service.chat(req.session_id, req.message)
    return ChatResponse(**result)


@router.post("/stream")
async def chat_stream(req: ChatRequest):
    async def event_generator():
        async for chunk in chat_service.stream_chat(req.session_id, req.message):
            import json
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
