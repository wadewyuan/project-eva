import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.models.schemas import VoiceSynthesizeRequest, VoiceTranscribeResponse
from app.services.voice_service import voice_service

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
