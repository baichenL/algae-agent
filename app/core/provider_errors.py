from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FailureInfo:
    code: str
    category: str
    retryable: bool
    public_message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "category": self.category,
            "retryable": self.retryable,
            "public_message": self.public_message,
        }


def _failure(code: str, *, retryable: bool, public_message: str) -> dict[str, Any]:
    return FailureInfo(
        code=code,
        category=code,
        retryable=retryable,
        public_message=public_message,
    ).to_dict()


def safe_provider_error(exc: Exception) -> dict[str, Any]:
    message = str(exc).casefold()
    name = type(exc).__name__
    module = type(exc).__module__
    if name in {"ValidationError", "RequestValidationError"}:
        return _failure(
            "validation_error",
            retryable=False,
            public_message="请求数据未通过校验，请检查输入后重试。",
        )
    if name in {"ToolExecutionError", "ToolReturnedError", "ToolContractViolation"}:
        return _failure(
            "tool_error",
            retryable=False,
            public_message="工具调用失败，未完成对应操作。",
        )
    if any(
        marker in message
        for marker in (
            "supported api model names",
            "model_not_found",
            "model `",
            "invalid_request_error",
        )
    ):
        return _failure(
            "model_configuration_error",
            retryable=False,
            public_message=(
                "\u6a21\u578b\u80fd\u529b\u914d\u7f6e\u4e0e\u5f53\u524d"
                "\u670d\u52a1\u63a5\u53e3\u4e0d\u517c\u5bb9\uff0c"
                "\u8bf7\u8054\u7cfb\u7ba1\u7406\u5458\u68c0\u67e5\u6a21\u578b\u6ce8\u518c\u8868\u3002"
            ),
        )
    provider_exception = module.startswith(("openai", "httpx", "httpcore")) or name in {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "APIStatusError",
        "InternalServerError",
    }
    if provider_exception or any(
        marker in message
        for marker in ("timeout", "temporarily unavailable", "rate limit", "429", "503")
    ):
        return _failure(
            "provider_error",
            retryable=True,
            public_message="模型或外部服务暂时不可用，可稍后重试。",
        )
    return _failure(
        "application_error",
        retryable=False,
        public_message="应用处理失败，系统未执行任何未经授权的外部操作。",
    )
