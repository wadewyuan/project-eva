import asyncio
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from app.models.schemas import VoiceSynthesizeRequest, VoiceTranscribeResponse
from app.services.voice_service import (
    STREAMING_PARTIAL_INTERVAL_SEC,
    voice_service,
)
from app.core.persona_engine import persona_engine

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


@router.get("/capabilities")
async def get_capabilities():
    """Advertise current TTS provider capabilities so the frontend can choose
    between the streaming WebSocket path and the bulk HTTP path."""
    return {
        "provider": voice_service.tts_provider,
        "streaming": voice_service.tts_supports_streaming,
        "media_type": voice_service.tts_media_type,
    }


@router.post("/synthesize")
async def synthesize_text(req: VoiceSynthesizeRequest):
    try:
        active_persona = persona_engine.get_active()
        voice = active_persona.get("voice")
        ref_wav = active_persona.get("voxcpm_ref_wav")
        prompt_text = active_persona.get("voxcpm_prompt_text")

        audio_data = await voice_service.synthesize_bytes(
            req.text,
            voice=voice,
            ref_wav=ref_wav,
            prompt_text=prompt_text,
        )

        media_type = voice_service.tts_media_type
        filename = f"speech{voice_service.tts_output_suffix}"

        async def data_generator():
            yield audio_data

        return StreamingResponse(
            data_generator(),
            media_type=media_type,
            headers={
                "Content-Disposition": f"inline; filename={filename}",
                "Content-Length": str(len(audio_data)),
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


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


# ---------------------------------------------------------------------------
# Real-time streaming TTS via WebSocket (VoxCPM only)
# ---------------------------------------------------------------------------

@router.websocket("/synthesize_stream")
async def websocket_synthesize(websocket: WebSocket):
    """
    Streaming TTS protocol.

    Client -> Server (text JSON):
      - {"text": "..."}  initial synthesis request

    Server -> Client:
      - {"type": "meta",  "sampleRate": int, "channels": 1}
      - Binary frames: raw PCM int16 LE mono at the advertised sample rate
      - {"type": "end"}   end of synthesis
      - {"type": "error", "message": "..."}
    """
    await websocket.accept()
    try:
        if not voice_service.tts_supports_streaming:
            await websocket.send_json({
                "type": "error",
                "message": "Streaming TTS not supported by current provider",
            })
            await websocket.close()
            return

        try:
            init_msg = await websocket.receive_json()
        except Exception:
            await websocket.close()
            return

        text = (init_msg or {}).get("text", "").strip()
        if not text:
            await websocket.send_json({"type": "error", "message": "empty text"})
            await websocket.close()
            return

        try:
            sample_rate = voice_service.voxcpm_sample_rate
        except Exception as exc:
            await websocket.send_json({"type": "error", "message": str(exc)})
            await websocket.close()
            return

        await websocket.send_json({
            "type": "meta",
            "sampleRate": sample_rate,
            "channels": 1,
        })

        active_persona = persona_engine.get_active()
        ref_wav = active_persona.get("voxcpm_ref_wav")
        prompt_text = active_persona.get("voxcpm_prompt_text")

        try:
            async for pcm_chunk in voice_service.synthesize_streaming_pcm(
                text,
                ref_wav=ref_wav,
                prompt_text=prompt_text,
            ):
                await websocket.send_bytes(pcm_chunk)
            await websocket.send_json({"type": "end"})
        except Exception as exc:
            try:
                await websocket.send_json({"type": "error", "message": str(exc)})
            except Exception:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
