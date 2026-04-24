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
- `app/services/voice_service.py` — ASR uses local `Qwen3-ASR-0.6B` on CUDA via `qwen_asr`. TTS uses `edge-tts` (cloud Azure). The ASR model is lazy-loaded on first use.

### Configuration

- `config/settings.py` — Pydantic-settings, reads from `.env`. Key vars: `LLM_BASE_URL`, `LLM_MODEL`, `DB_PATH`, `MAX_CONTEXT_MESSAGES`.
- `config/default_persona.yaml` — Default persona (小夏). Defines tones, speaking styles, personality traits, and forbidden phrases.
- `personas/` — Additional personas and per-tone few-shot examples.

### Voice pipeline

- **ASR:** `POST /voice/transcribe` uploads audio → `VoiceService.transcribe()` loads Qwen3-ASR on CUDA.
- **TTS:** `POST /voice/synthesize` streams MP3 from `edge-tts`.
- The frontend implements sentence-buffered auto-play TTS during streaming chat.

### Frontend

- Single-page app in `app/static/index.html` (no build step). Served at `/` and `/static`.
- Uses SSE for streaming chat, MediaRecorder for voice input, and queued audio playback for TTS.

## External Dependencies

- A local LLM server must be running at `LLM_BASE_URL` (default `http://localhost:8080/v1`). This is **not** bundled; the user runs it separately (e.g., llama.cpp, vLLM, Ollama).
- ASR requires the Qwen3-ASR model weights in `models/Qwen3-ASR-0.6B/` and a CUDA GPU.
- TTS requires internet access for `edge-tts` (Azure TTS).
