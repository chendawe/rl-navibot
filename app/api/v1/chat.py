from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from app.services.llm_service import llm_stream_think

router = APIRouter()

@router.post("/api/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    return StreamingResponse(
        llm_stream_think(data.get("msg", "")), 
        media_type="text/plain; charset=utf-8"
    )
