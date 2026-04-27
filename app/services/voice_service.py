import asyncio
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import edge_tts
import numpy as np
import torch
from qwen_asr import Qwen3ASRModel

MODEL_PATH = os.environ.get(
    "QWEN_ASR_MODEL_PATH",
    os.path.expanduser("~/src/project-eva/models/Qwen3-ASR-0.6B/")
)
TTS_VOICE = "zh-CN-XiaoxiaoNeural"

# Real-time streaming ASR parameters
STREAMING_SR = 16000
STREAMING_PARTIAL_INTERVAL_SEC = 1.0
STREAMING_PARTIAL_WINDOW_SEC = 8.0
STREAMING_MIN_PARTIAL_SEC = 1.0
STREAMING_SESSION_TTL_SEC = 300

_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"   # emoticons
    "\U0001F300-\U0001F5FF"   # symbols & pictographs
    "\U0001F680-\U0001F6FF"   # transport & map symbols
    "\U0001F1E0-\U0001F1FF"   # flags
    "\U00002702-\U000027B0"   # dingbats
    "\U0001F900-\U0001F9FF"   # supplemental symbols
    "\U0001FA00-\U0001FA6F"   # chess, etc.
    "\U0001FA70-\U0001FAFF"   # symbols & pictographs ext-a
    "☀-⛿"            # misc symbols
    "✀-➿"            # dingbats
    "‍"                   # zwj
    "️"                   # variation selector-16
    "]+",
    flags=re.UNICODE,
)


def _strip_emoji(text: str) -> str:
    return _EMOJI_RE.sub("", text).strip()


@dataclass
class StreamingSession:
    """Per-WebSocket streaming session state."""

    session_id: str
    audio: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    last_partial_audio_len: int = 0
    is_finalizing: bool = False
    last_activity: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_activity = time.time()


class VoiceService:
    def __init__(self) -> None:
        self._model = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._streaming_sessions: Dict[str, StreamingSession] = {}

    def _load_model(self) -> Qwen3ASRModel:
        if self._model is None:
            self._model = Qwen3ASRModel.from_pretrained(
                MODEL_PATH,
                dtype=torch.bfloat16,
                device_map="cuda:0",
                max_new_tokens=256,
            )
        return self._model

    # ------------------------------------------------------------------
    # Non-streaming transcription
    # ------------------------------------------------------------------

    async def transcribe(self, audio_path: Path) -> str:
        model = self._load_model()

        def _do_transcribe():
            results = model.transcribe(
                audio=str(audio_path),
                language=None,
            )
            return results[0].text if results else ""

        text = await asyncio.get_event_loop().run_in_executor(self._executor, _do_transcribe)
        return text.strip()

    # ------------------------------------------------------------------
    # Streaming transcription (real-time)
    # ------------------------------------------------------------------

    def create_streaming_session(self) -> StreamingSession:
        sid = uuid.uuid4().hex
        session = StreamingSession(session_id=sid)
        self._streaming_sessions[sid] = session
        return session

    def get_streaming_session(self, session_id: str) -> StreamingSession | None:
        session = self._streaming_sessions.get(session_id)
        if session:
            session.touch()
        return session

    def remove_streaming_session(self, session_id: str) -> None:
        self._streaming_sessions.pop(session_id, None)

    def gc_streaming_sessions(self) -> None:
        now = time.time()
        dead = [
            sid for sid, s in self._streaming_sessions.items()
            if now - s.last_activity > STREAMING_SESSION_TTL_SEC
        ]
        for sid in dead:
            self.remove_streaming_session(sid)

    def append_streaming_audio(self, session: StreamingSession, audio_pcm16: bytes) -> None:
        """Append raw PCM16 bytes to a streaming session's audio buffer."""
        if not audio_pcm16:
            return
        wav = np.frombuffer(audio_pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        session.audio = np.concatenate([session.audio, wav])
        session.touch()

    async def try_streaming_partial(self, session: StreamingSession) -> str | None:
        """Emit a partial transcription if enough new audio has arrived."""
        if session.is_finalizing or len(session.audio) == 0:
            return None

        new_audio = len(session.audio) - session.last_partial_audio_len
        min_samples = int(STREAMING_MIN_PARTIAL_SEC * STREAMING_SR)
        if new_audio < min_samples:
            return None

        model = self._load_model()

        # Transcribe the last N seconds to keep latency bounded.
        window_samples = int(STREAMING_PARTIAL_WINDOW_SEC * STREAMING_SR)
        if len(session.audio) > window_samples:
            window = session.audio[-window_samples:]
        else:
            window = session.audio

        def _do():
            results = model.transcribe(audio=(window, STREAMING_SR), language=None)
            return results[0].text if results else ""

        text = await asyncio.get_event_loop().run_in_executor(self._executor, _do)
        session.last_partial_audio_len = len(session.audio)
        session.touch()
        return text.strip() if text else None

    async def finish_streaming(self, session: StreamingSession) -> str:
        """Transcribe all accumulated audio and remove the session."""
        session.is_finalizing = True
        if len(session.audio) == 0:
            self.remove_streaming_session(session.session_id)
            return ""

        model = self._load_model()

        def _do():
            results = model.transcribe(audio=(session.audio, STREAMING_SR), language=None)
            return results[0].text if results else ""

        text = await asyncio.get_event_loop().run_in_executor(self._executor, _do)
        self.remove_streaming_session(session.session_id)
        return text.strip()

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------

    async def synthesize(self, text: str, output_path: Path) -> None:
        text = _strip_emoji(text)
        if not text:
            raise ValueError("Text is empty after removing emojis")
        communicate = edge_tts.Communicate(text, voice=TTS_VOICE)
        await communicate.save(str(output_path))


voice_service = VoiceService()
