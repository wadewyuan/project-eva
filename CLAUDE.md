# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Eva (小夏) is a FastAPI-based emotional companion chatbot. It features tone-aware persona switching, long-term memory, and voice input/output. The default persona is a best-friend-style Chinese companion.

## Common Commands

- **Run the server:** `uv run uvicorn app.main:app --reload --port 8000`
- **Install dependencies:** `uv sync`
- **Add a dependency:** `uv add <package>`
- **Run with a specific Python version:** Project uses Python 3.11 (see `.python-version`).

There are no tests, lint, or format commands configured yet.

## Architecture

### High-level request flow

1. `app/api/chat.py` receives a chat request (REST or SSE stream).
2. `app/services/chat_service.py` orchestrates the pipeline:
   - Detect tone via `llm_client.detect_tone()` + keyword regex overrides (`_TUCAO_KEYWORDS`).
   - Fetch relevant memories from SQLite via `memory_engine`.
   - Build a dynamic system prompt via `persona_engine.build_system_prompt(tone, memories)`.
   - Call the LLM with `N` recent messages as context (`settings.max_context_messages`).
   - Save the exchange and fire-and-forget memory extraction (`llm_client.extract_memories`).
3. `app/core/llm_client.py` talks to an OpenAI-compatible API (default: local server at `localhost:8080`).

### Key modules

- `app/core/llm_client.py` — LLM client, tone detection, and memory extraction prompts. All three are LLM calls.
- `app/core/memory_engine.py` — Async SQLite via `aiosqlite`. Stores sessions, messages, and memories. No vector search; `get_relevant_memories` returns the most recent 10 memories for the session.
- `app/core/persona_engine.py` — Loads personas from YAML. The active persona's system prompt is built dynamically from config + tone-specific few-shot examples (loaded from `personas/{id}/examples/{tone}.yaml`) + memories.
- `app/services/voice_service.py` — ASR uses local `Qwen3-ASR-0.6B` on CUDA via `qwen_asr`. TTS uses `edge-tts` (cloud Azure). The ASR model is lazy-loaded on first use. `StreamingSession` manages per-WebSocket audio buffers with bounded-window partial transcription (last 8s) and full-audio final transcription.

### Configuration

- `config/settings.py` — Pydantic-settings, reads from `.env`. Key vars: `LLM_BASE_URL`, `LLM_MODEL`, `DB_PATH`, `MAX_CONTEXT_MESSAGES`.
- `config/default_persona.yaml` — Default persona (小夏). Defines tones, speaking styles, personality traits, and forbidden phrases.
- `personas/` — Additional personas and per-tone few-shot examples.

### Voice pipeline

- **ASR (file):** `POST /voice/transcribe` uploads audio → `VoiceService.transcribe()` loads Qwen3-ASR on CUDA.
- **ASR (streaming):** `WS /voice/stream` receives PCM16 chunks and returns partial/final transcripts. See [Real-time streaming ASR](#real-time-streaming-asr) below.
- **TTS:** `POST /voice/synthesize` streams MP3 from `edge-tts`.
- The frontend implements sentence-buffered auto-play TTS during streaming chat.

### Real-time streaming ASR

The streaming pipeline uses a single WebSocket connection per utterance:

1. Frontend detects speech via energy-based VAD.
2. On speech start, opens `WS /voice/stream`.
3. Frontend resamples mic audio to 16kHz, converts float32 → PCM16, and sends binary chunks every 250ms.
4. Backend accumulates audio into a per-session buffer.
5. A background task emits partial transcriptions every 1s by transcribing the **last 8 seconds** of accumulated audio (bounded window, keeps latency constant).
6. On silence timeout, frontend sends text `"final"`; backend transcribes the **full accumulated audio** and returns the final result.
7. WebSocket closes; auto-submit triggers the chat flow.

Key design decisions:
- Uses the same `Qwen3ASRModel.from_pretrained()` instance for both streaming and file transcription (no separate service).
- `ThreadPoolExecutor(max_workers=1)` serializes all GPU inference. This is fine for MVP / 1-3 concurrent users.
- The windowing strategy is **backend-agnostic**; the frontend protocol does not change if we swap the inference backend later.

### Multi-user scaling roadmap

Current state is optimized for MVP simplicity, not concurrent load.

| Phase | Users | Approach | Effort |
|-------|-------|----------|--------|
| **Now** | 1-3 | Single worker, Transformers backend. | — |
| **Phase 1** | 3-10 | Increase `ThreadPoolExecutor(max_workers=N)` where N fits in GPU memory. Each 0.6B model instance in bfloat16 uses ~2-3GB. An 8-12GB GPU can run 2-4 concurrent inferences. Limited by Python GIL + GPU kernel launch overhead. | Small |
| **Phase 2** | 10-50 | **Batched inference.** Collect pending partial/final requests over a small time window (~50ms) and call `model.transcribe(audio=[(wav1, sr), (wav2, sr), ...])`. Qwen3ASRModel supports native batched audio input. This gives most of vLLM's throughput benefit without the service complexity. | Medium |
| **Phase 3** | 50+ | **vLLM backend swap.** Keep the WebSocket + windowing frontend unchanged. Swap `Qwen3ASRModel.from_pretrained()` for `Qwen3ASRModel.LLM()` in `VoiceService._load_model()`. Add a config flag (e.g., `ASR_BACKEND=vllm`). vLLM's continuous batching and PagedAttention become worthwhile at this scale. | Medium |

**Migration principle:** The frontend (WebSocket protocol, VAD, PCM16 streaming) and the windowing strategy (8s partial window, full-audio final) do not change across phases. Only the inference backend changes.

### Frontend

- Single-page app in `app/static/index.html` (no build step). Served at `/` and `/static`.
- Uses SSE for streaming chat, MediaRecorder for voice input, and queued audio playback for TTS.

## External Dependencies

- A local LLM server must be running at `LLM_BASE_URL` (default `http://localhost:8080/v1`). This is **not** bundled; the user runs it separately (e.g., llama.cpp, vLLM, Ollama).
- ASR requires the Qwen3-ASR model weights in `models/Qwen3-ASR-0.6B/` and a CUDA GPU.
- TTS requires internet access for `edge-tts` (Azure TTS).
