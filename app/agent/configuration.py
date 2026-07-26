# app/agent/configuration.py
from dataclasses import dataclass, field
from typing import Any, Dict

from app.core.model_registry import model_name

@dataclass(kw_only=True)
class AgentConfiguration:
    """定义运行时可动态调整的智能体超参数"""
    model_name: str = field(default_factory=lambda: model_name("agent"))
    temperature: float = 0.1
    max_tokens: int = 2048

    @classmethod
    def from_runnable_config(cls, config: Dict[str, Any]) -> "AgentConfiguration":
        """从 LangGraph 的 configurable 属性中提取参数"""
        configurable = config.get("configurable", {})
        return cls(
            model_name=configurable.get("model_name", model_name("agent")),
            temperature=configurable.get("temperature", 0.1),
            max_tokens=configurable.get("max_tokens", 2048),
        )
