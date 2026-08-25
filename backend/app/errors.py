"""Shared error payload shape.

Every failure surfaced to the UI carries the same four fields so the interface
can always show an explanation, a likely cause and a recommended action instead
of a bare technical string. `detail` is the only part the user has to opt into
by opening the technical details panel.
"""
from __future__ import annotations

from typing import Any


def error_payload(
    code: str,
    title: str,
    message: str,
    cause: str,
    action: str,
    detail: str = "",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "code": code,
        "title": title,
        "message": message,
        "cause": cause,
        "action": action,
        "detail": detail,
        **extra,
    }
