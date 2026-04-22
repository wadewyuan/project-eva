from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import chat, memory, persona
from app.core.memory_engine import memory_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    await memory_engine.init_db()
    yield


app = FastAPI(title="Eva - 虚拟情感陪伴助手", lifespan=lifespan)

app.include_router(chat.router)
app.include_router(memory.router)
app.include_router(persona.router)

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    index_path = static_dir / "index.html"
    if index_path.exists():
        from fastapi.responses import FileResponse
        return FileResponse(str(index_path))
    return {"message": "Eva API is running"}


@app.get("/health")
async def health():
    return {"status": "ok"}
