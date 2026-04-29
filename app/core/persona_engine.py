import os
from pathlib import Path

import yaml

from config.settings import settings


class PersonaEngine:
    def __init__(self):
        self.personas_dir = Path(settings.personas_dir)
        self.personas_dir.mkdir(parents=True, exist_ok=True)
        self.active_persona_id: str | None = None
        self._cache: dict[str, dict] = {}
        self._load_default()

    def _load_default(self) -> None:
        default_path = Path(settings.default_persona_path)
        if default_path.exists():
            with open(default_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            self._cache["default"] = data
            self.active_persona_id = "default"

    def load_persona(self, persona_id: str) -> dict | None:
        if persona_id in self._cache:
            return self._cache[persona_id]

        path = self.personas_dir / f"{persona_id}.yaml"
        if not path.exists():
            return None

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self._cache[persona_id] = data
        return data

    def list_personas(self) -> list[dict]:
        result = []
        for pid, data in self._cache.items():
            result.append({
                "id": pid,
                "name": data.get("name", pid),
                "description": data.get("description", ""),
                "is_active": pid == self.active_persona_id,
            })

        for f in self.personas_dir.glob("*.yaml"):
            pid = f.stem
            if pid not in self._cache:
                with open(f, "r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh)
                self._cache[pid] = data
                result.append({
                    "id": pid,
                    "name": data.get("name", pid),
                    "description": data.get("description", ""),
                    "is_active": pid == self.active_persona_id,
                })
        return result

    def set_active(self, persona_id: str) -> bool:
        if self.load_persona(persona_id):
            self.active_persona_id = persona_id
            return True
        return False

    def get_active(self) -> dict:
        pid = self.active_persona_id or "default"
        return self.load_persona(pid) or self._cache.get("default", {})

    def _load_tone_examples(self, persona_id: str, tone: str) -> list[dict]:
        """Load few-shot examples for a tone from external file, fallback to inline config."""
        examples_dir = self.personas_dir / persona_id / "examples"
        file_path = examples_dir / f"{tone}.yaml"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, list):
                return data

        # Fallback: inline few_shot_examples in persona YAML
        persona = self._cache.get(persona_id) or self.load_persona(persona_id)
        if not persona:
            return []
        few_shot_map = persona.get("few_shot_examples", {})
        if isinstance(few_shot_map, dict):
            return few_shot_map.get(tone, few_shot_map.get("default", []))
        elif isinstance(few_shot_map, list):
            return few_shot_map
        return []

    def build_system_prompt(self, tone: str = "default", memories: list[str] | None = None) -> str:
        persona = self.get_active()
        persona_id = self.active_persona_id or "default"

        name = persona.get("name", "AI助手")
        role = persona.get("role", "情感陪伴助手")
        description = persona.get("description", "")

        tone_map = persona.get("tone", {})
        selected_tone = tone_map.get(tone, tone_map.get("default", "温暖、自然"))

        styles = persona.get("speaking_style", [])
        traits = persona.get("personality_traits", [])
        forbidden = persona.get("forbidden", [])

        lines = [
            f"你是{name}，{role}。{description}",
            "",
            f"【当前语气】{selected_tone}",
            "",
            "【说话风格】",
        ]
        for s in styles:
            lines.append(f"- {s}")

        if traits:
            lines.append("")
            lines.append("【性格特点】")
            for t in traits:
                lines.append(f"- {t}")

        lines.append("")
        lines.append("【回复原则】")
        lines.append("- 回复要简短自然，像真实聊天。通常控制在2-3句话以内，除非用户明确要求详细解释")
        lines.append("- 不要长篇大论、不要总结分析、不要说教给建议")
        lines.append("- 用户说得多的时候多听少说，简单回应表示在听就好，不用过度展开")
        lines.append("- 不要每句结尾都强行提问，允许自然冷场")
        lines.append("- 聊天是双向的，给用户留说话空间")

        global_forbidden = [
            "禁止写超过3句话的回复（用户要求详细说明除外）",
            "禁止分析、总结、解读用户情绪",
            "禁止说教、给人生建议、灌鸡汤",
            "禁止连续追问多个问题",
            "禁止每句话结尾都加反问",
        ]
        lines.append("")
        lines.append("【禁忌】")
        for f in forbidden:
            lines.append(f"- {f}")
        for f in global_forbidden:
            lines.append(f"- {f}")

        # Few-shot examples — loaded from external files, fallback to inline
        few_shots = self._load_tone_examples(persona_id, tone) or self._load_tone_examples(persona_id, "default")
        if few_shots:
            lines.append("")
            lines.append("【对话风格示例】")
            for i, ex in enumerate(few_shots, 1):
                lines.append(f"示例 {i}:")
                lines.append(f"用户：{ex['user']}")
                lines.append(f"你：{ex['assistant']}")
                lines.append("")

        if memories:
            lines.append("")
            lines.append("【你记得关于用户的事】")
            for m in memories:
                lines.append(f"- {m}")

        return "\n".join(lines)

    def save_persona(self, persona_id: str, data: dict) -> None:
        path = self.personas_dir / f"{persona_id}.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False)
        self._cache[persona_id] = data


persona_engine = PersonaEngine()
