from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence


def _split_keys(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def resolve_gemini_keys(api_keys: str | Sequence[str] | None = None) -> list[str]:
    if isinstance(api_keys, str):
        candidates = _split_keys(api_keys)
    elif api_keys:
        candidates = []
        for item in api_keys:
            candidates.extend(_split_keys(str(item)))
    else:
        env_value = (
            os.getenv("GEMINI_API_KEYS")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or ""
        )
        candidates = _split_keys(env_value)

    seen: set[str] = set()
    resolved: list[str] = []
    for key in candidates:
        if key and key not in seen:
            seen.add(key)
            resolved.append(key)
    return resolved


def is_gemini_quota_error(exc: Exception) -> bool:
    message = f"{type(exc).__name__}: {exc}".upper()
    return any(
        token in message
        for token in (
            "429",
            "RESOURCE_EXHAUSTED",
            "RATE LIMIT",
            "RATE_LIMIT",
            "TOO MANY REQUESTS",
            "QUOTA",
        )
    )


@dataclass
class GeminiKeyRotator:
    keys: list[str]
    index: int = 0

    def __post_init__(self) -> None:
        if not self.keys:
            raise ValueError("At least one Gemini API key is required.")

    @property
    def current_key(self) -> str:
        return self.keys[self.index]

    @property
    def size(self) -> int:
        return len(self.keys)

    def can_rotate(self) -> bool:
        return self.size > 1

    def rotate(self) -> str:
        self.index = (self.index + 1) % self.size
        return self.current_key

    def masked_current_key(self) -> str:
        key = self.current_key
        if len(key) <= 8:
            return "*" * len(key)
        return f"{key[:4]}...{key[-4:]}"
