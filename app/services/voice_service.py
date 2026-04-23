import asyncio
import os
from pathlib import Path

import edge_tts
import torch
from qwen_asr import Qwen3ASRModel

MODEL_PATH = os.environ.get(
    "QWEN_ASR_MODEL_PATH",
    os.path.expanduser("~/src/project-eva/models/Qwen3-ASR-0.6B/")
)
TTS_VOICE = "zh-CN-XiaoxiaoNeural"


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
        communicate = edge_tts.Communicate(text, voice=TTS_VOICE)
        await communicate.save(str(output_path))


voice_service = VoiceService()
