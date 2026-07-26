from __future__ import annotations

import asyncio
import os
import signal

from app.core.database import init_db
from app.services.approval_execution_service import recover_approved_executions
from app.services.notifications.scheduler_service import email_scheduler_loop


async def run_worker() -> None:
    os.environ["TRUSTED_WORKER_PROCESS"] = "true"
    init_db()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):
            pass

    scheduler = asyncio.create_task(email_scheduler_loop())
    try:
        while not stop.is_set():
            await recover_approved_executions()
            try:
                await asyncio.wait_for(stop.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
    finally:
        scheduler.cancel()
        try:
            await scheduler
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(run_worker())
