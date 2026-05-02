import asyncio
import io
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, Dict

import edge_tts
import numpy as np
import soundfile as sf
import torch
from edge_tts.exceptions import NoAudioReceived
from qwen_asr import Qwen3ASRModel

from config.settings import settings

MODEL_PATH = os.environ.get(
    "QWEN_ASR_MODEL_PATH",
    os.path.expanduser("~/src/project-eva/models/Qwen3-ASR-0.6B/")
)
TTS_VOICE = "zh-CN-XiaoxiaoNeural"
VOXCPM_MODEL_ID = "openbmb/VoxCPM2"

# VoxCPM optional import
try:
    from voxcpm import VoxCPM
except ImportError:
    VoxCPM = None  # type: ignore[misc,assignment]

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
    """Per-WebSocket streaming session state.

    Audio chunks are stored in a list to avoid O(n²) copying from repeated
    np.concatenate on long utterances. Concatenation happens only at inference time.
    """

    session_id: str
    audio_chunks: list = field(default_factory=list)  # list[np.ndarray]
    total_samples: int = 0
    last_partial_total_samples: int = 0
    is_finalizing: bool = False
    last_activity: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_activity = time.time()


class VoiceService:
    def __init__(self) -> None:
        self._model = None
        # Single executor for all GPU work (ASR + TTS). Sharing one queue prevents
        # ASR partials from being blocked behind a long TTS generation, since both
        # contend for the same GPU anyway.
        self._executor = ThreadPoolExecutor(max_workers=settings.asr_max_workers)
        self._streaming_sessions: Dict[str, StreamingSession] = {}

        self._voxcpm_model = None
        self._voxcpm_lock = threading.Lock()

    def _load_model(self) -> Qwen3ASRModel:
        if self._model is None:
            self._model = Qwen3ASRModel.from_pretrained(
                MODEL_PATH,
                dtype=torch.bfloat16,
                device_map="cuda:0",
                max_new_tokens=256,
            )
        return self._model

    def _load_voxcpm_model(self):
        # Double-checked locking: fast path avoids the lock once the model is loaded.
        if self._voxcpm_model is not None:
            return self._voxcpm_model
        with self._voxcpm_lock:
            if self._voxcpm_model is not None:
                return self._voxcpm_model
            if VoxCPM is None:
                raise RuntimeError(
                    "voxcpm is not installed. Run: uv add voxcpm soundfile"
                )

            # VoxCPM hard-codes torch.compile(mode="reduce-overhead"), which enables
            # CUDA Graphs. CUDA Graphs key state to thread-local storage and crash
            # with `_is_key_in_tls` assertions when the model is invoked from a
            # ThreadPoolExecutor worker. Downgrade the mode at the torch.compile
            # boundary so we keep Inductor codegen but skip graph capture.
            import torch
            if not getattr(torch.compile, "_eva_patched", False):
                _orig_compile = torch.compile
                def _safe_compile(*args, **kwargs):
                    if kwargs.get("mode") == "reduce-overhead":
                        kwargs["mode"] = "default"
                    return _orig_compile(*args, **kwargs)
                _safe_compile._eva_patched = True  # type: ignore[attr-defined]
                torch.compile = _safe_compile  # type: ignore[assignment]

            os.environ.setdefault("HF_ENDPOINT", settings.tts_voxcpm_hf_endpoint)
            if settings.tts_voxcpm_model_source == "modelscope":
                from modelscope import snapshot_download
                model_dir = snapshot_download(
                    "openbmb/VoxCPM2",
                    cache_dir=settings.tts_voxcpm_cache_dir,
                )
                model = VoxCPM(
                    voxcpm_model_path=model_dir,
                    zipenhancer_model_path=None,
                    enable_denoiser=False,
                    optimize=True,
                )
            else:
                model = VoxCPM.from_pretrained(
                    VOXCPM_MODEL_ID,
                    load_denoiser=settings.tts_voxcpm_load_denoiser,
                    cache_dir=settings.tts_voxcpm_cache_dir,
                    optimize=True,
                )

            # Warmup: pay torch.compile + autotuning cost now, on the loader thread,
            # so the first user-facing request doesn't eat 5–15s of JIT.
            try:
                model.generate(
                    text="初始化",
                    cfg_value=settings.tts_voxcpm_cfg_value,
                    inference_timesteps=settings.tts_voxcpm_inference_timesteps,
                    normalize=True,
                )
            except Exception:
                pass

            self._voxcpm_model = model
        return self._voxcpm_model

    @property
    def tts_provider(self) -> str:
        return settings.tts_provider

    @property
    def tts_output_suffix(self) -> str:
        # Both providers return MP3: edge natively, voxcpm encoded from PCM.
        return ".mp3"

    @property
    def tts_media_type(self) -> str:
        return "audio/mpeg"

    @property
    def tts_supports_streaming(self) -> bool:
        return settings.tts_provider == "voxcpm"

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
        """Append raw PCM16 bytes to a streaming session's audio buffer.

        Chunks are accumulated in a list; concatenation only happens at inference time
        to avoid O(n²) copying on long utterances.
        """
        if not audio_pcm16:
            return
        wav = np.frombuffer(audio_pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        session.audio_chunks.append(wav)
        session.total_samples += len(wav)
        session.touch()

    @staticmethod
    def _get_audio(session: StreamingSession) -> np.ndarray:
        """Concatenate all chunks once. O(total) instead of O(n²)."""
        if not session.audio_chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(session.audio_chunks)

    async def try_streaming_partial(self, session: StreamingSession) -> str | None:
        """Emit a partial transcription if enough new audio has arrived."""
        if session.is_finalizing or session.total_samples == 0:
            return None

        new_samples = session.total_samples - session.last_partial_total_samples
        min_samples = int(STREAMING_MIN_PARTIAL_SEC * STREAMING_SR)
        if new_samples < min_samples:
            return None

        model = self._load_model()
        audio = self._get_audio(session)

        # Transcribe the last N seconds to keep latency bounded.
        window_samples = int(STREAMING_PARTIAL_WINDOW_SEC * STREAMING_SR)
        if len(audio) > window_samples:
            window = audio[-window_samples:]
        else:
            window = audio

        def _do():
            results = model.transcribe(audio=(window, STREAMING_SR), language=None)
            return results[0].text if results else ""

        text = await asyncio.get_event_loop().run_in_executor(self._executor, _do)
        session.last_partial_total_samples = session.total_samples
        session.touch()
        return text.strip() if text else None

    async def finish_streaming(self, session: StreamingSession) -> str:
        """Transcribe all accumulated audio and remove the session."""
        session.is_finalizing = True
        if session.total_samples == 0:
            self.remove_streaming_session(session.session_id)
            return ""

        model = self._load_model()
        audio = self._get_audio(session)

        def _do():
            results = model.transcribe(audio=(audio, STREAMING_SR), language=None)
            return results[0].text if results else ""

        text = await asyncio.get_event_loop().run_in_executor(self._executor, _do)
        self.remove_streaming_session(session.session_id)
        return text.strip()

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------

    async def synthesize_bytes(
        self,
        text: str,
        voice: str | None = None,
        ref_wav: str | None = None,
        prompt_text: str | None = None,
    ) -> bytes:
        """Synthesize `text` and return the encoded audio as MP3 bytes.

        Both providers return MP3:
        - edge_tts: streamed natively as MP3, collected into memory.
        - voxcpm: PCM float32 → int16 → MP3 via libsndfile (no disk roundtrip).

        For VoxCPM, `ref_wav` + `prompt_text` clone the voice from a reference
        audio so output stays consistent across calls. Without them VoxCPM
        samples a fresh speaker each time.
        """
        text = _strip_emoji(text)
        if not text:
            raise ValueError("Text is empty after removing emojis")

        if settings.tts_provider == "voxcpm":
            return await self._synthesize_voxcpm_mp3(text, ref_wav, prompt_text)
        return await self._synthesize_edge_mp3(text, voice=voice)

    async def _synthesize_edge_mp3(self, text: str, voice: str | None = None) -> bytes:
        voice_to_use = voice if voice else TTS_VOICE
        last_error = None
        for attempt in range(2):
            try:
                communicate = edge_tts.Communicate(text, voice=voice_to_use)
                buf = bytearray()
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        buf.extend(chunk["data"])
                if not buf:
                    raise NoAudioReceived("empty audio stream")
                return bytes(buf)
            except NoAudioReceived as e:
                last_error = e
                if attempt == 0:
                    await asyncio.sleep(0.5)
                continue

        raise RuntimeError(f"Edge TTS synthesis failed after retries: {last_error}")

    @staticmethod
    def _resolve_voxcpm_ref(ref_wav: str | None) -> str | None:
        """Resolve a possibly-relative ref_wav path to an absolute path.

        Returns None if the path is missing or doesn't exist (caller falls back
        to random-speaker synthesis).
        """
        if not ref_wav:
            return None
        path = Path(ref_wav)
        if not path.is_absolute():
            path = Path(os.getcwd()) / path
        return str(path) if path.exists() else None

    def _build_voxcpm_kwargs(
        self,
        text: str,
        ref_wav: str | None,
        prompt_text: str | None,
    ) -> dict:
        kwargs: dict = {
            "text": text,
            "cfg_value": settings.tts_voxcpm_cfg_value,
            "inference_timesteps": settings.tts_voxcpm_inference_timesteps,
            "normalize": True,
        }
        # Fall through: persona config -> global default -> random speaker.
        resolved_ref = (
            self._resolve_voxcpm_ref(ref_wav)
            or self._resolve_voxcpm_ref(settings.tts_voxcpm_ref_wav)
        )
        resolved_prompt = prompt_text or settings.tts_voxcpm_prompt_text
        if resolved_ref and resolved_prompt:
            kwargs["prompt_wav_path"] = resolved_ref
            kwargs["prompt_text"] = resolved_prompt
        return kwargs

    async def _synthesize_voxcpm_mp3(
        self,
        text: str,
        ref_wav: str | None = None,
        prompt_text: str | None = None,
    ) -> bytes:
        model = self._load_voxcpm_model()
        kwargs = self._build_voxcpm_kwargs(text, ref_wav, prompt_text)

        def _do_generate() -> bytes:
            wav = model.generate(**kwargs)
            buf = io.BytesIO()
            sf.write(buf, wav, model.tts_model.sample_rate, format="MP3")
            return buf.getvalue()

        return await asyncio.get_event_loop().run_in_executor(self._executor, _do_generate)

    @property
    def voxcpm_sample_rate(self) -> int:
        """Sample rate of the loaded VoxCPM model. Loads the model if needed."""
        return self._load_voxcpm_model().tts_model.sample_rate

    async def synthesize_streaming_pcm(
        self,
        text: str,
        ref_wav: str | None = None,
        prompt_text: str | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """Stream VoxCPM audio as raw PCM int16 LE chunks (mono, model sample rate).

        Each yielded value is a `bytes` object containing PCM frames ready to
        ship over a WebSocket. Use `voxcpm_sample_rate` for the rate.

        For VoxCPM, `ref_wav` + `prompt_text` clone the voice from a reference
        audio so output stays consistent across calls. Without them VoxCPM
        samples a fresh speaker each time.

        Raises:
            RuntimeError: if the current provider is not voxcpm.
        """
        text = _strip_emoji(text)
        if not text:
            raise ValueError("Text is empty after removing emojis")
        if settings.tts_provider != "voxcpm":
            raise RuntimeError("Streaming TTS only supported with VoxCPM provider")

        model = self._load_voxcpm_model()
        kwargs = self._build_voxcpm_kwargs(text, ref_wav, prompt_text)
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _run():
            try:
                for chunk in model.generate_streaming(**kwargs):
                    pcm = np.clip(chunk, -1.0, 1.0)
                    pcm_i16 = (pcm * 32767.0).astype("<i2").tobytes()
                    loop.call_soon_threadsafe(queue.put_nowait, pcm_i16)
                loop.call_soon_threadsafe(queue.put_nowait, None)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)

        self._executor.submit(_run)

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item


voice_service = VoiceService()
