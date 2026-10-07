"""运行记录落盘 —— `.bear/multiagents/<session_id>/`。

为什么单独一个模块：落盘和"跑编排"是两个独立的变化原因。
编排只负责算出结果；记录放哪、什么时候建目录、写到一半崩了怎么办，
是另一套要求，塞在 runner.py 里会让 runner 承担两个变更来源。

目录形状：

    .bear/multiagents/
    └── <session_id>/
        ├── runs.jsonl                  每次运行追加一行摘要
        └── run-20261007-121733.json    单次运行的完整记录

三条硬要求：

1. **先建目录，再跑图。** 由调用方（interface.execute_multiagent）在开跑前就叫
   `session_dir()`。这样中途崩溃或被中断时，目录和已经发生的事实都还在。
2. **原子写。** 先写 `.tmp` 再 `os.replace`，避免中断留下半个 JSON。
3. **失败也要写。** 编排抛异常时写一条 `status=failed` 的记录再往上抛。
   否则目录里只有成功记录，出问题时什么都查不到。

`session_id` 为空时退化到 `_manual` 子目录并如实记录，不静默丢弃。

注意：这里的目录是**运行记录**，与 `agents/subagent.py` 读的
`.bear/subagent/`（自定义子代理**定义**）是两回事，别混。
"""
from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RECORDS_DIRNAME = "multiagents"
MANUAL_SESSION = "_manual"
INDEX_FILENAME = "runs.jsonl"
_SKIP_SUMMARY = {"events", "results", "tasks", "verdicts", "intercepts", "final_answer"}


def records_root(cwd: Path | str | None = None) -> Path:
    """`.bear/multiagents` 的绝对路径。不创建目录。"""
    return Path(cwd or Path.cwd()) / ".bear" / RECORDS_DIRNAME


def session_dir(session_id: str | None, *, cwd: Path | str | None = None,
                root: Path | str | None = None) -> Path:
    """返回某次会话的记录目录并确保它存在。

    `root` 只给测试用（指向 tmp 目录），正常路径不传。
    """
    base = Path(root) if root is not None else records_root(cwd)
    d = base / (session_id or MANUAL_SESSION)
    d.mkdir(parents=True, exist_ok=True)
    return d


def summarise(record: dict[str, Any]) -> dict[str, Any]:
    """给 runs.jsonl 用的一行摘要。刻意不含正文，行要短。"""
    s = record.get("summary") or {}
    return {
        "ts": record.get("started_at"),
        "session_id": record.get("session_id"),
        "status": record.get("status", "ok"),
        "task": record.get("query"),
        "wall_seconds": record.get("wall_seconds"),
        "tokens": s.get("tokens_used"),
        "completeness": s.get("completeness"),
        "iteration": s.get("iteration"),
        "dispatches": s.get("dispatches"),
        "stop_reason": s.get("stop_reason"),
        "unfinished": len([v for v in record.get("verdicts", []) if v.get("status") != "complete"]),
        "contradictions": sum(len(v.get("contradictions") or []) for v in record.get("verdicts", [])),
        "invariant_violations": record.get("invariant_violations") or [],
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    os.replace(tmp, path)


def _append_index(directory: Path, entry: dict[str, Any]) -> None:
    line = json.dumps(entry, ensure_ascii=False, default=str)
    with (directory / INDEX_FILENAME).open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def save_run(record: dict[str, Any], *, session_id: str | None = None,
             cwd: Path | str | None = None, root: Path | str | None = None) -> Path:
    """写一次成功运行的完整记录 + 追加一行索引。返回记录文件路径。"""
    directory = session_dir(session_id or record.get("session_id"), cwd=cwd, root=root)
    record = {**record, "session_id": session_id or record.get("session_id"),
              "status": record.get("status", "ok")}

    path = directory / f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    # 同一秒内跑两次会撞名（测试里常见）：加毫秒后缀，仍然原子写
    if path.exists():
        path = directory / f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}.json"

    _atomic_write_json(path, record)
    _append_index(directory, {**summarise(record), "record": path.name})
    return path


def save_failure(*, session_id: str | None, task: str, error: BaseException,
                 wall_seconds: float, budget: Any = None,
                 cwd: Path | str | None = None, root: Path | str | None = None) -> Path:
    """编排自己抛异常时也要留一条记录，否则记录目录里只有成功案例。"""
    directory = session_dir(session_id, cwd=cwd, root=root)
    record: dict[str, Any] = {
        "status": "failed",
        "session_id": session_id,
        "query": task,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_seconds": round(wall_seconds, 2),
        "error": f"{type(error).__name__}: {error}",
        "traceback": traceback.format_exc(),
    }
    if budget is not None:
        record["budget"] = {
            "max_concurrent": getattr(budget, "max_concurrent", None),
            "max_iterations": getattr(budget, "max_iterations", None),
            "token_budget": getattr(budget, "token_budget", None),
            "enable_purification": getattr(budget, "enable_purification", None),
        }

    path = directory / f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}-failed.json"
    if path.exists():
        path = directory / (f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                            f"-{int(time.time() * 1000) % 1000:03d}-failed.json")
    _atomic_write_json(path, record)
    _append_index(directory, {**summarise(record), "record": path.name, "error": record["error"]})
    return path


def load_index(session_id: str | None, *, cwd: Path | str | None = None,
               root: Path | str | None = None) -> list[dict[str, Any]]:
    """读回该会话的索引行。文件不存在就返回空列表，不抛。"""
    path = session_dir(session_id, cwd=cwd, root=root) / INDEX_FILENAME
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


__all__ = ["MANUAL_SESSION", "INDEX_FILENAME", "records_root", "session_dir",
           "summarise", "save_run", "save_failure", "load_index"]
