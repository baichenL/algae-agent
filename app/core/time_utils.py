import datetime
import os
from zoneinfo import ZoneInfo


APP_TIMEZONE = os.getenv("APP_TIMEZONE", "Asia/Shanghai")


def local_now() -> datetime.datetime:
    return datetime.datetime.now(ZoneInfo(APP_TIMEZONE))


def local_time_string(value: datetime.datetime | None = None) -> str:
    current = value or local_now()
    if current.tzinfo is not None:
        current = current.astimezone(ZoneInfo(APP_TIMEZONE))
    return current.strftime("%Y-%m-%d %H:%M:%S")


def local_iso_string(value: datetime.datetime | None = None) -> str:
    current = value or local_now()
    if current.tzinfo is not None:
        current = current.astimezone(ZoneInfo(APP_TIMEZONE))
    return current.replace(microsecond=0).isoformat()


def time_days_ago(days: int) -> str:
    return local_time_string(local_now() - datetime.timedelta(days=max(int(days or 0), 0)))


def parse_time(value: str | None) -> datetime.datetime | None:
    if not value:
        return None

    text = value.strip()
    parsers = [
        lambda item: datetime.datetime.fromisoformat(item.replace("Z", "+00:00")),
        lambda item: datetime.datetime.strptime(item, "%Y-%m-%d %H:%M:%S"),
        lambda item: datetime.datetime.strptime(item, "%Y-%m-%d"),
    ]
    for parser in parsers:
        try:
            parsed = parser(text)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=ZoneInfo(APP_TIMEZONE))
            return parsed.astimezone(ZoneInfo(APP_TIMEZONE))
        except (TypeError, ValueError):
            continue
    return None


def days_since_time(value: str | None, fallback: int = 0) -> int:
    parsed = parse_time(value)
    if not parsed:
        return int(fallback or 0)
    return max((local_now().date() - parsed.date()).days, 0)


def display_time_string(value: str | None) -> str | None:
    if not value:
        return None

    text = str(value).strip()
    if "T" in text:
        try:
            parsed = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=datetime.timezone.utc)
            return local_time_string(parsed)
        except ValueError:
            return text

    parsed = parse_time(text)
    return local_time_string(parsed) if parsed else text
