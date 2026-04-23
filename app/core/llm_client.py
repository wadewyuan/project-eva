import json
from typing import AsyncIterator

from openai import AsyncOpenAI

from config.settings import settings


class LLMClient:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )
        self.model = settings.llm_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens

    async def chat(self, messages: list[dict[str, str]], stream: bool = False) -> str | AsyncIterator[str]:
        """Call the LLM. Returns full text or an async iterator for streaming."""
        if stream:
            return self._stream_chat(messages)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        msg = response.choices[0].message
        return (msg.content or "").strip()

    async def _stream_chat(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None) or ""
                if text:
                    yield text

    async def detect_tone(self, user_message: str) -> str:
        """Lightweight tone/scene detection."""
        prompt = (
            "判断以下用户消息的对话场景类型。只输出一个标签，不要解释。\n\n"
            "场景定义：\n"
            "- 吐槽：抱怨、发泄不满、骂人、说'烦/有病/不想待了/垃圾'等\n"
            "- 八卦：传递或打听他人隐私、传闻、关系动态\n"
            "- 倾诉烦恼：表达低落、累、难过、困惑，需要被倾听\n"
            "- 分享喜悦：分享好消息、开心的事、成就\n"
            "- 默认：日常闲聊、陈述事实、问问题等不属于以上四类的情况\n\n"
            "示例：\n"
            "'我老板真烦，天天让我加班' → 吐槽\n"
            "'这破公司我是一天也不想待了' → 吐槽\n"
            "'房东又涨房租了，有病吧' → 吐槽\n"
            "'听说小李跟他对象分手了' → 八卦\n"
            "'你知道吗，隔壁在裁员' → 八卦\n"
            "'我最近压力好大，睡不着' → 倾诉烦恼\n"
            "'我感觉什么都做不好' → 倾诉烦恼\n"
            "'我考上研究生了！' → 分享喜悦\n"
            "'咪咪今天蹭我了' → 分享喜悦\n"
            "'今天吃什么好' → 默认\n"
            "'我养了只猫叫咪咪' → 默认\n\n"
            f"用户消息：{user_message}\n"
            "标签："
        )
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )
        msg = response.choices[0].message
        tone = (msg.content or "").strip() or "默认"
        tone = tone.strip().replace("，", "").replace("。", "").replace("：", "").replace(" ", "")
        tone_map = {
            "吐槽": "tucao",
            "八卦": "gossip",
            "倾诉烦恼": "venting",
            "分享喜悦": "sharing_joy",
        }
        if tone in tone_map:
            return tone_map[tone]
        return "default"

    async def extract_memories(self, user_msg: str, assistant_reply: str) -> list[dict]:
        """Extract 0-3 important facts from a conversation turn."""
        # Truncate assistant reply to avoid overwhelming the model
        short_reply = assistant_reply[:200] if len(assistant_reply) > 200 else assistant_reply
        prompt = (
            "从以下用户消息中提取值得长期记住的重要信息（如名字、喜好、工作、重要事件等）。\n"
            "每行一个，格式：事实内容 | 分类\n"
            "分类可选：喜好、生活、关系、工作、情绪、其他\n"
            "如果没有则只回复：无\n"
            "只输出列表，不要解释。\n\n"
            f"用户消息：{user_msg}\n"
            "提取结果："
        )
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1024,
            )
            msg = response.choices[0].message
            raw = (msg.content or "").strip()
            if not raw or raw == "无":
                return []
            results = []
            for line in raw.split("\n"):
                line = line.strip()
                if not line or line == "无":
                    continue
                if "|" in line:
                    parts = line.split("|", 1)
                    fact = parts[0].strip().lstrip("-").strip()
                    category = parts[1].strip()
                    if fact:
                        results.append({"fact": fact, "category": category})
                else:
                    results.append({"fact": line.lstrip("-").strip(), "category": "其他"})
            return results[:3]
        except Exception:
            pass
        return []


llm_client = LLMClient()
