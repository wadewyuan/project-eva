import asyncio
import tempfile
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

from app.models.schemas import VoiceSynthesizeRequest, VoiceTranscribeResponse
from app.services.voice_service import (
    STREAMING_PARTIAL_INTERVAL_SEC,
    voice_service,
)

router = APIRouter(prefix="/voice", tags=["voice"])


@router.post("/transcribe", response_model=VoiceTranscribeResponse)
async def transcribe_audio(audio: UploadFile = File(...)):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(audio.filename or "audio.webm").suffix
        ) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        text = await voice_service.transcribe(tmp_path)
        return VoiceTranscribeResponse(text=text)
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()


@router.post("/synthesize")
async def synthesize_text(req: VoiceSynthesizeRequest):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp_path = Path(tmp.name)

        await voice_service.synthesize(req.text, tmp_path)

        def file_generator():
            with open(tmp_path, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    yield chunk
            if tmp_path.exists():
                tmp_path.unlink()

        return StreamingResponse(
            file_generator(),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "inline; filename=speech.mp3",
            },
        )
    except ValueError as exc:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()
        raise


# ---------------------------------------------------------------------------
# Real-time streaming ASR via WebSocket
# ---------------------------------------------------------------------------

@router.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket protocol for real-time streaming ASR.

    Client -> Server:
      - Binary message: raw PCM16 mono 16kHz audio bytes
      - Text message "final": request final transcription

    Server -> Client:
      - {"type": "partial", "text": "..."}  (incremental result)
      - {"type": "final",  "text": "..."}  (accurate final result)
      - {"type": "error",  "message": "..."}
    """
    await websocket.accept()
    session = voice_service.create_streaming_session()

    async def _partial_emitter():
        """Background task: emit partial transcriptions on an interval."""
        try:
            while True:
                await asyncio.sleep(STREAMING_PARTIAL_INTERVAL_SEC)
                s = voice_service.get_streaming_session(session.session_id)
                if s is None or s.is_finalizing:
                    return
                text = await voice_service.try_streaming_partial(s)
                if text:
                    await websocket.send_json({"type": "partial", "text": text})
        except asyncio.CancelledError:
            return
        except Exception:
            return

    emitter_task = asyncio.create_task(_partial_emitter())

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.receive":
                if "bytes" in message and message["bytes"]:
                    voice_service.append_streaming_audio(session, message["bytes"])
                elif "text" in message and message["text"] == "final":
                    break
    except WebSocketDisconnect:
        pass
    finally:
        emitter_task.cancel()
        try:
            await emitter_task
        except asyncio.CancelledError:
            pass

        try:
            text = await voice_service.finish_streaming(session)
            await websocket.send_json({"type": "final", "text": text})
        except Exception as exc:
            await websocket.send_json({"type": "error", "message": str(exc)})
        finally:
            try:
                await websocket.close()
            except Exception:
                pass
