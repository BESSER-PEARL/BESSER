"""
Temporary file cleanup service for the BESSER backend.

Scans the system temp directory for BESSER-created temporary directories
and removes those older than a configurable threshold. A periodic background
task ensures cleanup runs automatically while the application is running.
"""

import asyncio
import logging
import os
import shutil
import tempfile
import time

logger = logging.getLogger(__name__)

# All known prefixes used by BESSER when creating temp directories.
# Keep this in sync with constants.py and any ad-hoc prefixes in routers.
_BESSER_TEMP_PREFIXES = (
    "besser_",
    "besser_agent_",
    "besser_csv_",
    "besser_llm_",
    "user_profile_",
)

# How often the background cleanup task runs (in seconds).
_CLEANUP_INTERVAL_SECONDS = 3600  # 1 hour


def cleanup_old_temp_files(max_age_hours: int = 24) -> None:
    """Remove BESSER-created temp directories older than *max_age_hours*.

    The function iterates over the system temp directory, identifies
    directories whose names start with one of the known BESSER prefixes,
    and deletes any that were last modified more than *max_age_hours* ago.

    Errors on individual directories are logged and silently skipped so
    that one problematic entry does not prevent the rest from being cleaned.
    """
    tmp_root = tempfile.gettempdir()
    cutoff = time.time() - (max_age_hours * 3600)
    removed = 0

    try:
        entries = os.listdir(tmp_root)
    except OSError:
        logger.warning("Unable to list temp directory %s", tmp_root)
        return

    for entry in entries:
        if not any(entry.startswith(prefix) for prefix in _BESSER_TEMP_PREFIXES):
            continue

        full_path = os.path.join(tmp_root, entry)
        if not os.path.isdir(full_path):
            continue

        try:
            mtime = os.path.getmtime(full_path)
            if mtime < cutoff:
                shutil.rmtree(full_path, ignore_errors=True)
                removed += 1
        except OSError as exc:
            logger.debug("Skipping temp entry %s: %s", full_path, exc)

    if removed:
        logger.info("Cleaned up %d stale BESSER temp directories", removed)


async def _periodic_cleanup(max_age_hours: int = 24) -> None:
    """Run :func:`cleanup_old_temp_files` in a loop every hour."""
    while True:
        await asyncio.sleep(_CLEANUP_INTERVAL_SECONDS)
        try:
            cleanup_old_temp_files(max_age_hours=max_age_hours)
        except Exception:
            logger.exception("Error during periodic temp-file cleanup")


def schedule_cleanup(max_age_hours: int = 24) -> asyncio.Task:
    """Start the periodic cleanup background task.

    Should be called once during application startup (inside the lifespan
    context manager). Returns the :class:`asyncio.Task` so that the caller
    can cancel it on shutdown if desired.
    """
    task = asyncio.create_task(_periodic_cleanup(max_age_hours=max_age_hours))
    # Prevent the task from being garbage-collected while it is running.
    task.add_done_callback(lambda t: t.result() if not t.cancelled() else None)
    logger.info(
        "Scheduled periodic temp-file cleanup (max_age_hours=%d, interval=%ds)",
        max_age_hours,
        _CLEANUP_INTERVAL_SECONDS,
    )
    return task
