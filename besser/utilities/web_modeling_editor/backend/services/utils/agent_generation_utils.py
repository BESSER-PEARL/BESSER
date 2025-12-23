"""Utility helpers for building agent configuration bundles."""

from __future__ import annotations

import io
import re
import zipfile
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

AgentZipGenerator = Callable[[Any, Optional[Dict[str, Any]]], Tuple[io.BytesIO, str]]


def slugify_name(value: Optional[str], fallback: str) -> str:
    """Sanitize folder names while keeping them human readable."""
    name = (value or fallback or "").strip()
    safe_name = re.sub(r'[<>:"/\\|?*]+', "_", name)
    safe_name = re.sub(r"_+", "_", safe_name).strip("_ ")
    return safe_name or fallback


def append_zip_contents(
    target_zip: zipfile.ZipFile,
    source_buffer: io.BytesIO,
    prefix: Optional[str] = None,
) -> None:
    """Append contents of an in-memory zip archive into another archive."""
    source_buffer.seek(0)
    with zipfile.ZipFile(source_buffer, "r") as source_zip:
        for info in source_zip.infolist():
            arcname = info.filename
            if prefix:
                arcname = f"{prefix}/{arcname}" if arcname else prefix
            if info.is_dir():
                if not arcname.endswith('/'):
                    arcname += '/'
                target_zip.writestr(arcname, b"")
                continue
            target_zip.writestr(arcname, source_zip.read(info.filename))


def build_configurations_package(
    agent_model: Any,
    configurations: List[Dict[str, Any]],
    generate_agent_zip_fn: AgentZipGenerator,
) -> io.BytesIO:
    """Create a zip containing the base agent plus variants for each configuration."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as combined_zip:
        base_buffer, _ = generate_agent_zip_fn(agent_model, None)
        append_zip_contents(combined_zip, base_buffer)

        used_prefixes: Set[str] = set()
        for index, entry in enumerate(configurations):
            configuration_payload = entry.get("configuration")
            if not isinstance(configuration_payload, dict):
                continue

            sanitized_payload = dict(configuration_payload)
            sanitized_payload.pop("configurations", None)

            fallback_name = f"configuration_{index + 1}"
            folder_name = slugify_name(entry.get("name"), fallback_name)
            unique_name = folder_name
            suffix = 1
            while unique_name in used_prefixes:
                unique_name = f"{folder_name}_{suffix}"
                suffix += 1
            used_prefixes.add(unique_name)

            config_buffer, _ = generate_agent_zip_fn(agent_model, sanitized_payload)
            append_zip_contents(combined_zip, config_buffer, prefix=unique_name)

    zip_buffer.seek(0)
    return zip_buffer
