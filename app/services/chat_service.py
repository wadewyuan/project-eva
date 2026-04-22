import re
import uuid

from app.core.llm_client import llm_client
from app.core.memory_engine import memory_engine
from app.core.persona_engine import persona_engine
from config.settings import settings

# Keywords that strongly indicate complaint/venting tone
_TUCAO_KEYWORDS = re.compile(
    r'烦|讨厌|垃圾|破公司|有病|恶心|气死|受不了|不想待|剥削|压榨|'
    r'加班|老板.*(让|叫|逼)|推.*活|推给|涨房租|扣钱|迟到.*扣',
    re.UNICODE,
)
_TOPIC_SWITCH_KEYWORDS = re.compile(
    r'不说这个|换个话题|聊点别的|不说他了|不说她了|先别说',
    re.UNICODE,
)


def _maybe_override_tone(message: str, detected: str) -> str:
    """If detect_tone returns default but message smells like a complaint, force tucao."""
    if detected == "default" and _TUCAO_KEYWORDS.search(message):
        return "tucao"
    return detected


def _is_topic_switch(message: str) -> bool:
    """Detect if user is explicitly changing topic."""
    return bool(_TOPIC_SWITCH_KEYWORDS.search(message))


class ChatService:
    async def chat(self, session_id: str | None, message: str) -> dict:
        if not session_id:
            session_id = str(uuid.uuid4())
            await memory_engine.create_session(session_id, title=message[:20])

        # Ensure session exists
        await memory_engine.create_session(session_id)

        # 1. Detect tone
        tone = await llm_client.detect_tone(message)
        tone = _maybe_override_tone(message, tone)
        # If current detection is default, check if recent session history was tucao
        # But skip if user is clearly switching topic
        if tone == "default" and not _is_topic_switch(message):
            recent_msgs = await memory_engine.get_session_messages(session_id, limit=6)
            for m in recent_msgs:
                if _maybe_override_tone(m["content"], "default") == "tucao":
                    tone = "tucao"
                    break

        # 2. Fetch memories
        memories = await memory_engine.get_relevant_memories(session_id, message)

        # 3. Build system prompt
        system_prompt = persona_engine.build_system_prompt(tone=tone, memories=memories)

        # 4. Fetch recent messages for context
        recent = await memory_engine.get_session_messages(session_id, limit=settings.max_context_messages)
        messages = [{"role": "system", "content": system_prompt}]
        for m in recent:
            messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": message})

        # 5. Call LLM
        reply = await llm_client.chat(messages, stream=False)

        # 6. Save messages
        await memory_engine.add_message(session_id, "user", message)
        await memory_engine.add_message(session_id, "assistant", reply)
        await memory_engine.update_session_time(session_id)

        # 7. Async extract memories (MVP: fire and forget, could be async task later)
        try:
            extracted = await llm_client.extract_memories(message, reply)
            for item in extracted:
                fact = item.get("fact", "")
                category = item.get("category", "其他")
                if fact:
                    await memory_engine.add_memory(
                        session_id=session_id,
                        fact_text=fact,
                        category=category,
                        importance_score=5,
                    )
        except Exception:
            pass

        return {
            "session_id": session_id,
            "reply": reply,
            "tone_detected": tone,
        }

    async def stream_chat(self, session_id: str | None, message: str):
        """Generator for SSE streaming. Yields dict chunks."""
        if not session_id:
            session_id = str(uuid.uuid4())
            await memory_engine.create_session(session_id, title=message[:20])

        await memory_engine.create_session(session_id)

        tone = await llm_client.detect_tone(message)
        tone = _maybe_override_tone(message, tone)
        if tone == "default":
            recent_msgs = await memory_engine.get_session_messages(session_id, limit=6)
            for m in recent_msgs:
                if _maybe_override_tone(m["content"], "default") == "tucao":
                    tone = "tucao"
                    break

        memories = await memory_engine.get_relevant_memories(session_id, message)
        system_prompt = persona_engine.build_system_prompt(tone=tone, memories=memories)

        recent = await memory_engine.get_session_messages(session_id, limit=settings.max_context_messages)
        messages = [{"role": "system", "content": system_prompt}]
        for m in recent:
            messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": message})

        # Save user message immediately
        await memory_engine.add_message(session_id, "user", message)

        full_reply = ""
        async for chunk in await llm_client.chat(messages, stream=True):
            full_reply += chunk
            yield {"type": "token", "data": chunk}

        # Save assistant reply
        await memory_engine.add_message(session_id, "assistant", full_reply)
        await memory_engine.update_session_time(session_id)

        # Extract memories
        try:
            extracted = await llm_client.extract_memories(message, full_reply)
            for item in extracted:
                fact = item.get("fact", "")
                category = item.get("category", "其他")
                if fact:
                    await memory_engine.add_memory(
                        session_id=session_id,
                        fact_text=fact,
                        category=category,
                        importance_score=5,
                    )
        except Exception:
            pass

        yield {"type": "done", "session_id": session_id, "tone_detected": tone}


chat_service = ChatService()
