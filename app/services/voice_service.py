import asyncio
import os
import re
from pathlib import Path

import edge_tts
import torch
from qwen_asr import Qwen3ASRModel

MODEL_PATH = os.environ.get(
    "QWEN_ASR_MODEL_PATH",
    os.path.expanduser("~/src/project-eva/models/Qwen3-ASR-0.6B/")
)
TTS_VOICE = "zh-CN-XiaoxiaoNeural"

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


class VoiceService:
    def __init__(self) -> None:
        self._model = None

    def _load_model(self) -> Qwen3ASRModel:
        if self._model is None:
            self._model = Qwen3ASRModel.from_pretrained(
                MODEL_PATH,
                dtype=torch.bfloat16,
                device_map="cuda:0",
                max_new_tokens=256,
            )
        return self._model

    async def transcribe(self, audio_path: Path) -> str:
        model = await asyncio.get_event_loop().run_in_executor(
            None, self._load_model
        )

        def _do_transcribe():
            results = model.transcribe(
                audio=str(audio_path),
                language=None,  # auto language detection
            )
            return results[0].text if results else ""

        text = await asyncio.get_event_loop().run_in_executor(None, _do_transcribe)
        return text.strip()

    async def synthesize(self, text: str, output_path: Path) -> None:
        text = _strip_emoji(text)
        if not text:
            raise ValueError("Text is empty after removing emojis")
        communicate = edge_tts.Communicate(text, voice=TTS_VOICE)
        await communicate.save(str(output_path))


voice_service = VoiceService()
