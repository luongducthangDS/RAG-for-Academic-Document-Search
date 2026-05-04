"""
history.py
──────────
Lưu và tải lịch sử trò chuyện ra file JSON.
Mỗi session có một file riêng, đặt tên theo timestamp.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


HISTORY_DIR = Path(__file__).resolve().parent / "chat_history"
HISTORY_DIR.mkdir(exist_ok=True)


# ─── Định dạng tên file ───────────────────────────────────────────────────────

def _session_path(session_id: str) -> Path:
    return HISTORY_DIR / f"{session_id}.json"


def new_session_id() -> str:
    """Tạo session ID mới theo timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def list_sessions() -> list[dict]:
    """
    Trả về danh sách session đã lưu, sắp xếp mới nhất trước.
    Mỗi phần tử: {"id": "20240101_120000", "label": "01/01/2024 12:00", "path": Path}
    """
    sessions = []
    for f in sorted(HISTORY_DIR.glob("*.json"), reverse=True):
        try:
            ts = datetime.strptime(f.stem, "%Y%m%d_%H%M%S")
            label = ts.strftime("%d/%m/%Y %H:%M")
        except ValueError:
            label = f.stem
        sessions.append({"id": f.stem, "label": label, "path": f})
    return sessions


# ─── Đọc / Ghi ───────────────────────────────────────────────────────────────

def save_history(session_id: str, messages: list[dict]) -> None:
    """Ghi toàn bộ messages ra file JSON."""
    path = _session_path(session_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)


def load_history(session_id: str) -> list[dict]:
    """Đọc messages từ file JSON. Trả về [] nếu không tìm thấy."""
    path = _session_path(session_id)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def delete_session(session_id: str) -> None:
    """Xoá file lịch sử của một session."""
    path = _session_path(session_id)
    if path.exists():
        path.unlink()