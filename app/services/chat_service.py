import asyncio
import logging
import re
import uuid

from app.core.llm_client import llm_client
from app.core.memory_engine import memory_engine
from app.core.persona_engine import persona_engine
from config.settings import settings

logger = logging.getLogger(__name__)

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
    async def _prepare_chat(
        self,
        session_id: str | None,
        message: str,
    ) -> tuple[str, list[dict], str]:
        """Ensure session exists, detect tone, fetch memories, and build messages.

        Returns (session_id, messages_for_llm, tone_detected).
        """
        if not session_id:
            session_id = str(uuid.uuid4())
            await memory_engine.create_session(session_id, title=message[:20])

        await memory_engine.create_session(session_id)

        tone = await llm_client.detect_tone(message)
        tone = _maybe_override_tone(message, tone)
        if tone == "default" and not _is_topic_switch(message):
            recent_msgs = await memory_engine.get_session_messages(session_id, limit=6)
            for m in recent_msgs:
                if _maybe_override_tone(m["content"], "default") == "tucao":
                    tone = "tucao"
                    break

        profile = await memory_engine.get_user_profile()
        memories = await memory_engine.get_relevant_memories(session_id, message)
        system_prompt = persona_engine.build_system_prompt(tone=tone, profile=profile, memories=memories)

        recent = await memory_engine.get_session_messages(session_id, limit=settings.max_context_messages)
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        for m in recent:
            messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": message})

        return session_id, messages, tone

    async def _save_turn(
        self,
        session_id: str,
        assistant_reply: str,
    ) -> None:
        """Save assistant message and update session time."""
        await memory_engine.add_message(session_id, "assistant", assistant_reply)
        await memory_engine.update_session_time(session_id)

    async def _extract_memories_task(
        self,
        session_id: str,
        user_message: str,
        assistant_reply: str,
    ) -> None:
        """Background task: extract and store profile + experience memories."""
        source_context = f"用户：{user_message}\n助手：{assistant_reply}"
        try:
            extracted = await llm_client.extract_memories(user_message)

            # A. User profile (long-term facts)
            for p in extracted.get("profiles", []):
                fact_key = p.get("fact_key", "").strip()
                fact_value = p.get("fact_value", "").strip()
                if fact_key and fact_value:
                    await memory_engine.upsert_profile(
                        fact_type=p.get("fact_type", "other"),
                        fact_key=fact_key,
                        fact_value=fact_value,
                        confidence=p.get("importance", 5),
                        source_context=source_context,
                    )

            # B. Experience memories (time-bound events)
            for item in extracted.get("memories", []):
                fact = item.get("fact", "")
                category = item.get("category", "其他")
                importance = item.get("importance", 5)
                if fact:
                    await memory_engine.add_memory(
                        session_id=session_id,
                        fact_text=fact,
                        category=category,
                        importance_score=importance,
                        source_context=source_context,
                    )
        except Exception:
            logger.warning("Memory extraction failed", exc_info=True)

    def _fire_extract_memories(
        self,
        session_id: str,
        user_message: str,
        assistant_reply: str,
    ) -> None:
        """Fire-and-forget background memory extraction."""
        asyncio.create_task(
            self._extract_memories_task(session_id, user_message, assistant_reply)
        )

    async def chat(self, session_id: str | None, message: str) -> dict:
        session_id, messages, tone = await self._prepare_chat(session_id, message)

        reply = await llm_client.chat(messages, stream=False)
        assert isinstance(reply, str)

        await memory_engine.add_message(session_id, "user", message)
        await self._save_turn(session_id, reply)
        self._fire_extract_memories(session_id, message, reply)

        return {
            "session_id": session_id,
            "reply": reply,
            "tone_detected": tone,
        }

    async def stream_chat(self, session_id: str | None, message: str):
        """Generator for SSE streaming. Yields dict chunks."""
        session_id, messages, tone = await self._prepare_chat(session_id, message)

        # Save user message immediately so it's included in context on retry
        await memory_engine.add_message(session_id, "user", message)

        full_reply = ""
        async for chunk in await llm_client.chat(messages, stream=True):
            full_reply += chunk
            yield {"type": "token", "data": chunk}

        await self._save_turn(session_id, full_reply)
        yield {"type": "done", "session_id": session_id, "tone_detected": tone}
        self._fire_extract_memories(session_id, message, full_reply)


chat_service = ChatService()
