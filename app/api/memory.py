from fastapi import APIRouter

from app.core.memory_engine import memory_engine
from app.models.schemas import MemoryItem

router = APIRouter(prefix="/memories", tags=["memory"])


@router.get("", response_model=list[MemoryItem])
async def list_memories(session_id: str | None = None):
    rows = await memory_engine.list_memories(session_id)
    return [MemoryItem(**row) for row in rows]


@router.delete("/{memory_id}")
async def delete_memory(memory_id: int):
    await memory_engine.delete_memory(memory_id)
    return {"ok": True}


@router.delete("")
async def clear_all_memories():
    await memory_engine.clear_all_memories()
    return {"ok": True, "cleared": True}
