from fastapi import APIRouter

from app.core.persona_engine import persona_engine
from app.models.schemas import PersonaInfo

router = APIRouter(prefix="/personas", tags=["persona"])


@router.get("", response_model=list[PersonaInfo])
async def list_personas():
    return [PersonaInfo(**p) for p in persona_engine.list_personas()]


@router.put("/active")
async def set_active_persona(persona_id: str):
    ok = persona_engine.set_active(persona_id)
    return {"ok": ok, "active": persona_engine.active_persona_id}
