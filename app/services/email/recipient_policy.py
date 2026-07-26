from __future__ import annotations

import os


def get_default_recipients() -> list[str]:
    """Return the configured allowlist without loading the SMTP adapter."""
    raw = (
        os.getenv("EMAIL_ALLOWED_RECIPIENTS")
        or os.getenv("EMAIL_TO")
        or ""
    )
    return [item.strip() for item in raw.split(",") if item.strip()]
