"""Tier-1：带溯源链接的摘要索引。

一个单元 = 一段简洁摘要 + 一个稠密向量 + 一组指向 Tier-2 页号的溯源链接。
硬约束：**不存在没有溯源链接的单元**。这是「每条摘要都能回指原始证据」这条
性质在代码层面的落点。

本模块不含 LLM 调用：向量由调用方算好传进来，因此可以直接单测。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import config


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度。任一向量模长为 0 时返回 0，不抛异常。"""
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= config.COSINE_ZERO_EPS or norm_b <= config.COSINE_ZERO_EPS:
        return 0.0
    return dot / ((norm_a ** 0.5) * (norm_b ** 0.5))


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Unit:
    """一个摘要单元。``origin`` 区分它是按页生成的还是写回沉淀的。"""

    __slots__ = ("unit_id", "text", "embedding", "provenance", "created_at", "updated_at", "origin")

    def __init__(
        self,
        unit_id: str,
        text: str,
        embedding: list[float],
        provenance: list[str],
        origin: str = "ingest",
    ) -> None:
        self.unit_id = unit_id
        self.text = text
        self.embedding = embedding
        # 去重但保序：同一页被引用两次不应产生两条链接
        self.provenance = list(dict.fromkeys(provenance))
        self.created_at = _utc_now()
        self.updated_at = self.created_at
        self.origin = origin

    def to_dict(self) -> dict:
        return {
            "unit_id": self.unit_id,
            "text": self.text,
            "embedding": self.embedding,
            "provenance": self.provenance,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "origin": self.origin,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Unit":
        unit = cls(
            str(data.get("unit_id") or ""),
            str(data.get("text") or ""),
            list(data.get("embedding") or []),
            list(data.get("provenance") or []),
            str(data.get("origin") or "ingest"),
        )
        unit.created_at = str(data.get("created_at") or unit.created_at)
        unit.updated_at = str(data.get("updated_at") or unit.updated_at)
        return unit

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<Unit {self.unit_id} prov={self.provenance} origin={self.origin}>"


class Tier1Index:
    """摘要索引。序列化成一个 JSON 文件，规模到几千条都不需要外部向量库。"""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / config.TIER1_INDEX_FILENAME
        self._units: list[Unit] = []
        self._load()

    # ── 持久化 ─────────────────────────────────────────────
    def _load(self) -> None:
        self._units = []
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        rows = data.get("units") if isinstance(data, dict) else data
        if not isinstance(rows, list):
            return
        for row in rows:
            if isinstance(row, dict):
                unit = Unit.from_dict(row)
                if unit.unit_id and unit.provenance:
                    self._units.append(unit)

    def save(self) -> None:
        payload = {"units": [unit.to_dict() for unit in self._units]}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 写入 ───────────────────────────────────────────────
    def add(
        self,
        text: str,
        provenance: list[str],
        embedding: list[float],
        origin: str = "ingest",
    ) -> Unit:
        """新增一个单元。没有溯源链接直接拒绝，不给「无源摘要」留口子。"""
        if not provenance:
            raise ValueError("Tier-1 单元必须带溯源链接，拒绝写入无源摘要")
        unit_id = f"{config.UNIT_ID_PREFIX}{len(self._units) + 1:0{config.UNIT_ID_WIDTH}d}"
        unit = Unit(unit_id, text, embedding, provenance, origin)
        self._units.append(unit)
        self.save()
        return unit

    def update(
        self,
        unit_id: str,
        new_text: str,
        new_provenance: list[str],
        embedding: list[float],
    ) -> Unit | None:
        """就地更新并**继承**溯源：旧链接保留，新链接追加，形成完整审计链。"""
        for unit in self._units:
            if unit.unit_id != unit_id:
                continue
            unit.text = new_text
            unit.provenance = list(dict.fromkeys(unit.provenance + list(new_provenance)))
            unit.embedding = embedding
            unit.updated_at = _utc_now()
            self.save()
            return unit
        return None

    # ── 读取 ───────────────────────────────────────────────
    def get(self, unit_id: str) -> Unit | None:
        for unit in self._units:
            if unit.unit_id == unit_id:
                return unit
        return None

    def all(self) -> list[Unit]:
        return list(self._units)

    def count(self) -> int:
        return len(self._units)

    def search(
        self,
        vector: list[float],
        k: int,
        exclude: set[str] | None = None,
    ) -> list[tuple[Unit, float]]:
        """按向量相似度取 top-k。返回 ``(unit, score)``，按分数降序。"""
        if not self._units or not vector:
            return []
        scored: list[tuple[Unit, float]] = []
        for unit in self._units:
            if exclude and unit.unit_id in exclude:
                continue
            if not unit.embedding:
                continue
            scored.append((unit, cosine(vector, unit.embedding)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]
