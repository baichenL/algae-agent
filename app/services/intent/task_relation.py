from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskRelationSignals:
    cancel_previous: bool = False
    start_new: bool = False
    resume_named: bool = False
    evidence: tuple[str, ...] = ()


_CANCEL_PREVIOUS_PATTERNS = (
    re.compile(
        r"(?:不要|别|停止|取消)"
        r"(?:再|继续|处理|执行)?"
        r".{0,8}(?:刚才|之前|前一个|上一个)"
        r".{0,8}(?:任务|更新|操作|请求|流程)?"
    ),
    re.compile(
        r"(?:do\s+not\s+continue|stop|cancel)"
        r".{0,24}(?:previous|prior|last)"
        r".{0,16}(?:task|update|operation|request)?",
        re.I,
    ),
)
_START_NEW_PATTERNS = (
    re.compile(r"(?:开始|创建|发起)(?:一个)?新任务"),
    re.compile(r"(?:new\s+task|start\s+(?:a\s+)?new\s+task)\s*[:：]?", re.I),
)
_RESUME_NAMED_PATTERNS = (
    re.compile(r"(?:继续|恢复)(?:邮件|科学|调查|更新|传代)(?:任务|流程)?"),
    re.compile(
        r"resume.{0,16}(?:email|scientific|investigation|update|subculture).{0,8}task",
        re.I,
    ),
)


def parse_task_relation_signals(text: str) -> TaskRelationSignals:
    """Parse cross-turn task-control clauses independently from business intent.

    This grammar identifies the relationship between tasks; it never selects a
    route or capability by itself. Keeping it separate prevents each workflow
    from accumulating its own phrase list and disagreeing about lifecycle state.
    """

    normalized = " ".join(str(text or "").casefold().split())
    cancel_previous = any(pattern.search(normalized) for pattern in _CANCEL_PREVIOUS_PATTERNS)
    start_new = any(pattern.search(normalized) for pattern in _START_NEW_PATTERNS)
    resume_named = any(pattern.search(normalized) for pattern in _RESUME_NAMED_PATTERNS)
    evidence = tuple(
        label
        for matched, label in (
            (cancel_previous, "explicit_cancel_previous"),
            (start_new, "explicit_new_task"),
            (resume_named, "explicit_named_resume"),
        )
        if matched
    )
    return TaskRelationSignals(
        cancel_previous=cancel_previous,
        start_new=start_new,
        resume_named=resume_named,
        evidence=evidence,
    )
