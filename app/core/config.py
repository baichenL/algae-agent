# app/core/config.py
# 配置文件，包含系统参数、默认值和环境变量加载

import json
import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

LLM_API_KEY = (
    os.getenv("DEEPSEEK_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or "test-key"
)

client = OpenAI(
    api_key=LLM_API_KEY,
    base_url=os.getenv("DEEPSEEK_BASE_URL")
    or os.getenv("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com"),
)

MEMORY_DIR = "data/outputs"
MAX_SESSION_MESSAGES = int(os.getenv("MAX_SESSION_MESSAGES", "30"))
SYSTEM_PROMPT_VERSION = "2026-06-workflow-receipts-v1"

DEFAULT_SYSTEM_PROMPT = f"""你是一个微藻课题组内部实验室智能助手 Agent。

system_prompt_version={SYSTEM_PROMPT_VERSION}

核心原则：
1. 安全优先。真实写库、删除、硬件 workflow、传代执行都必须经过明确确认或 pending 审批。
2. 事实优先。实验室事实以数据库状态、context snapshot、工具返回结果为准；session history 只用于对话连续性。
3. 待审批动作不是已执行事实。pending action 只能表示有人提出了请求，不能当作数据库已经改变。
4. 不确定就追问。无法唯一确定 strain_id、操作对象、字段值或执行意图时，必须要求用户补充信息。
5. LLM 输出只能作为解释、建议或草稿，不直接代表实验记录或审批结果。
6. 普通 Chat 没有工具权限。没有结构化 pending_id、run_id 或 tool_result 时，禁止声称已调用工具、已创建请求或已执行操作。

回答实验室状态问题时，优先依据当前 DB snapshot 或工具结果。
提出新增、修改、删除品系时，只创建 pending 请求。
涉及硬件或传代 workflow 时，不要自行执行；应要求人工审批或明确确认后由系统安全路径处理。
"""

# 作用是清洗 session_id，确保文件名安全
def sanitize_session_id(session_id: str) -> str:
    safe_session_id = "".join(
        ch if ch.isalnum() or ch in "-_" else "_"
        for ch in str(session_id or "default_session")
    )
    return safe_session_id or "default_session"

# 作用是截断会话历史，确保不会超过最大消息数
def trim_conversation_history(conversation_history: list) -> list:
    if not isinstance(conversation_history, list):
        return []
    if MAX_SESSION_MESSAGES <= 0 or len(conversation_history) <= MAX_SESSION_MESSAGES:
        return conversation_history

    first_message = conversation_history[0] if conversation_history else None
    keep_system = isinstance(first_message, dict) and first_message.get("role") == "system"
    if keep_system and MAX_SESSION_MESSAGES <= 1:
        return [first_message]
    if keep_system:
        return [first_message] + conversation_history[-(MAX_SESSION_MESSAGES - 1):]
    return conversation_history[-MAX_SESSION_MESSAGES:]

# 作用是获取会话历史文件路径
def get_memory_file_path(session_id: str) -> str:
    return os.path.join(MEMORY_DIR, f"history_{sanitize_session_id(session_id)}.json")

# 作用是加载会话历史，如果不存在则返回默认系统提示
def load_memory(session_id: str) -> list:
    file_path = get_memory_file_path(session_id)
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                stored = json.load(f)
                dialogue = [
                    item for item in stored
                    if isinstance(item, dict) and item.get("role") in {"user", "assistant"}
                ]
                return [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}] + dialogue
        except Exception as exc:
            from app.services.observability.error_events import record_error_event

            record_error_event(
                session_id=session_id,
                layer="memory_db",
                component="config",
                operation="load_memory",
                severity="warning",
                error_type=type(exc).__name__,
                error_message=str(exc),
                metadata={"file_path": file_path},
            )
            pass
    return [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]


def save_memory(session_id: str, conversation_history: list):
    try:
        os.makedirs(MEMORY_DIR, exist_ok=True)
        file_path = get_memory_file_path(session_id)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(trim_conversation_history(conversation_history), f, ensure_ascii=False, indent=4)
    except Exception as exc:
        from app.services.observability.error_events import record_error_event

        record_error_event(
            session_id=session_id,
            layer="memory_db",
            component="config",
            operation="save_memory",
            severity="warning",
            error_type=type(exc).__name__,
            error_message=str(exc),
            metadata={"file_path": get_memory_file_path(session_id)},
        )
