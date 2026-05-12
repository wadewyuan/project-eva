import json
import re
from typing import AsyncIterator

from openai import AsyncOpenAI

from config.settings import settings

_THINKING_RE = re.compile(r"<(thinking|think)>.*?</\1>", re.DOTALL)


class LLMClient:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )
        self.model = settings.llm_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self._small_client: AsyncOpenAI | None = None

    @staticmethod
    def _strip_thinking(text: str) -> str:
        return _THINKING_RE.sub("", text).strip()

    @property
    def small_client(self) -> AsyncOpenAI:
        """Lazy-initialized small model client for tone detection."""
        if self._small_client is None:
            base_url = settings.llm_small_base_url or settings.llm_base_url
            api_key = settings.llm_small_api_key or settings.llm_api_key
            self._small_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        return self._small_client

    @property
    def small_model(self) -> str:
        """Small model name for tone detection. Falls back to main model if not set."""
        return settings.llm_small_model or settings.llm_model

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
        return self._strip_thinking(msg.content or "")

    async def _stream_chat(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        buffer = ""
        async for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None) or ""
                if not text:
                    continue
                buffer += text

                while True:
                    match = _THINKING_RE.search(buffer)
                    if not match:
                        break
                    before = buffer[:match.start()]
                    if before:
                        yield before
                    buffer = buffer[match.end():]

                if "<think" not in buffer and "<thinking" not in buffer:
                    if buffer:
                        yield buffer
                        buffer = ""

        if buffer:
            clean = re.sub(r"<(thinking|think)>.*", "", buffer, flags=re.DOTALL)
            clean = re.sub(r"</(thinking|think)>", "", clean)
            if clean:
                yield clean

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
        response = await self.small_client.chat.completions.create(
            model=self.small_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )
        msg = response.choices[0].message
        tone = self._strip_thinking(msg.content or "").strip() or "默认"
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

    async def extract_memories(self, user_msg: str) -> dict[str, list[dict]]:
        """Extract profile facts (A) and experience memories (B) from the user message.

        Returns {"profiles": [...], "memories": [...]}.
        """
        user_msg = self._strip_thinking(user_msg)
        prompt = (
            "从以下用户消息中提取两类信息，只输出列表，不要解释。\n\n"
            "【A类 - 用户画像】长期不变的事实：名字、职业、常住城市、"
            "固定喜好（如'喜欢猫'）、性格特点。这类信息应该能被当作'用户的属性'。\n"
            "格式：A | 事实类型 | 键 | 值 | 重要性(1-10)\n"
            "事实类型可选：identity, preference, career, relationship, habit, other\n"
            "键必须用英文 snake_case，如 name, job_title, hobby_pet\n"
            "示例：A | identity | name | 张三 | 10\n"
            "      A | preference | hobby | 喜欢跑步 | 6\n"
            "      A | career | job_title | 产品经理 | 7\n\n"
            "【B类 - 经历/事件】有时间属性或会过期的信息："
            "上周做了什么、下周计划、临时情绪、八卦、一次性的活动。\n"
            "格式：B | 内容 | 分类 | 时间属性 | 重要性(1-10)\n"
            "分类可选：喜好、生活、关系、工作、情绪、其他\n"
            "时间属性：past（已发生）/ future（未来计划）/ now（当前状态）/ unknown\n"
            "示例：B | 上周去了迪士尼 | 生活 | past | 7\n"
            "      B | 下周三要面试 | 工作 | future | 8\n"
            "      B | 最近压力很大 | 情绪 | now | 6\n\n"
            "如果没有值得记住的信息，只回复：无\n\n"
            f"用户消息：{user_msg}\n"
            "提取结果："
        )
        try:
            response = await self.small_client.chat.completions.create(
                model=self.small_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1024,
            )
            msg = response.choices[0].message
            raw = self._strip_thinking(msg.content or "")
            if not raw or raw == "无":
                return {"profiles": [], "memories": []}

            profiles: list[dict] = []
            memories: list[dict] = []

            for line in raw.split("\n"):
                line = line.strip()
                if not line or line == "无":
                    continue
                if "|" not in line:
                    continue

                parts = [p.strip() for p in line.split("|")]
                tag = parts[0].lstrip("-").strip().upper()

                if tag == "A" and len(parts) >= 4:
                    fact_type = parts[1]
                    fact_key = parts[2]
                    fact_value = parts[3]
                    importance = 5
                    if len(parts) > 4:
                        try:
                            importance = int(parts[4])
                            importance = max(1, min(10, importance))
                        except ValueError:
                            pass
                    if fact_key and fact_value:
                        profiles.append({
                            "fact_type": fact_type,
                            "fact_key": fact_key,
                            "fact_value": fact_value,
                            "importance": importance,
                        })

                elif tag == "B" and len(parts) >= 4:
                    content = parts[1]
                    category = parts[2]
                    time_attr = parts[3] if len(parts) > 3 else "unknown"
                    importance = 5
                    if len(parts) > 4:
                        try:
                            importance = int(parts[4])
                            importance = max(1, min(10, importance))
                        except ValueError:
                            pass
                    if content:
                        memories.append({
                            "fact": content,
                            "category": category,
                            "time_attr": time_attr,
                            "importance": importance,
                        })

            return {"profiles": profiles[:3], "memories": memories[:3]}
        except Exception:
            pass
        return {"profiles": [], "memories": []}


llm_client = LLMClient()
