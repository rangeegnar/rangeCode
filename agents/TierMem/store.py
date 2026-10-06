"""Tier-2：不可变分页原始日志。

页一旦写入就不再修改，并且拿到一个稳定编号。这是整个两级架构的地基：
Tier-1 的摘要要长期引用这些页号，所以页号绝不能因为后续追加而漂移。

设计上刻意不含任何 LLM 调用，也不含任何 async，纯本地文件操作。
"""

from __future__ import annotations

import json
from pathlib import Path

from . import config


def est_tokens(text: str) -> int:
    """粗略估算 token 数，只用于切页，不用于计费。

    中文按 ``TOKENS_PER_CJK_CHAR`` 计（默认 1 字 1 token），
    其余字符按 ``CHARS_PER_TOKEN`` 折算（默认 4 字符 1 token）。
    """
    cjk = 0
    other = 0
    for ch in text:
        if config.CJK_START <= ord(ch) <= config.CJK_END:
            cjk += 1
        else:
            other += 1
    return cjk * config.TOKENS_PER_CJK_CHAR + max(1, other // max(1, config.CHARS_PER_TOKEN))


def format_page_id(index: int) -> str:
    """按固定位宽生成页号，保证排序与引用在位数增长后依然稳定。"""
    return f"{config.PAGE_ID_PREFIX}{index:0{config.PAGE_ID_WIDTH}d}"


class Tier2Store:
    """append-only 的分页原始日志，落盘为一行一个 JSON 对象。"""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / config.TIER2_PAGES_FILENAME
        self._pages: list[dict] = []
        self._load()

    # ── 持久化 ─────────────────────────────────────────────
    def _load(self) -> None:
        self._pages = []
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # 坏行跳过而不是整份日志报废
                continue
            if isinstance(record, dict) and record.get("page_id"):
                self._pages.append(record)

    # ── 写入 ───────────────────────────────────────────────
    def append_pages(self, texts: list[str], session: str, created_at: str | None = None) -> list[str]:
        """追加若干页，返回新分配的页号。页号从当前总数继续往后排。"""
        new_ids: list[str] = []
        for text in texts:
            page_id = format_page_id(len(self._pages) + 1)
            record = {
                "page_id": page_id,
                "text": text,
                "session": session,
                "created_at": created_at or _utc_now(),
            }
            self._pages.append(record)
            new_ids.append(page_id)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return new_ids

    # ── 读取 ───────────────────────────────────────────────
    def get(self, page_id: str) -> dict | None:
        for page in self._pages:
            if page["page_id"] == page_id:
                return page
        return None

    def many(self, page_ids: list[str]) -> list[dict]:
        """按传入顺序返回页，忽略不存在的页号。"""
        index = {page["page_id"]: page for page in self._pages}
        return [index[pid] for pid in dict.fromkeys(page_ids) if pid in index]

    def all(self) -> list[dict]:
        return list(self._pages)

    def page_ids(self) -> list[str]:
        return [page["page_id"] for page in self._pages]

    def count(self) -> int:
        return len(self._pages)


def _utc_now() -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
