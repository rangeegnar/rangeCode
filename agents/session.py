
#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from typing import Any
import json

SESSION_DIRNAME = "sessions"


def get_project_session_dir() -> Path:
    """返回当前项目的会话存档目录，不存在时创建。"""
    d = Path.cwd() / ".range" / SESSION_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_session(session_id: str, data: dict[str, Any]) -> None:
    (get_project_session_dir() / f"{session_id}.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def load_session(session_id: str) -> dict[str, Any] | None:
    path = get_project_session_dir() / f"{session_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_sessions() -> list[dict[str, Any]]:
    results = []
    for f in get_project_session_dir().glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "metadata" in data:
            results.append(data["metadata"])
    return results


def get_latest_session_id() -> str | None:
    sessions = list_sessions()
    if not sessions:
        return None
    sessions.sort(key=lambda s: s.get("startTime", ""), reverse=True)
    return sessions[0].get("id")
