import datetime
import json
import os
from typing import List

from app.models.email_schema import EmailLogRecord


EMAIL_LOG_PATH = "data/logs/email_logs.jsonl"


def append_email_log(record: EmailLogRecord) -> str:
    log_id = record.created_at or datetime.datetime.utcnow().isoformat()
    os.makedirs(os.path.dirname(EMAIL_LOG_PATH), exist_ok=True)
    with open(EMAIL_LOG_PATH, "a", encoding="utf-8") as file:
        file.write(record.model_dump_json() + "\n")
    return log_id


def read_recent_email_logs(limit: int = 50) -> List[dict]:
    if not os.path.exists(EMAIL_LOG_PATH):
        return []
    with open(EMAIL_LOG_PATH, "r", encoding="utf-8") as file:
        lines = file.readlines()[-max(limit, 1):]
    logs = []
    for line in lines:
        try:
            logs.append(json.loads(line))
        except Exception:
            continue
    return list(reversed(logs))
