"""TierMem 编排器：摄入 → 充分性路由 → 有界升级 → 验证式写回。

对应论文 Algorithm 1。四个阶段各自独立，可以按 config 的开关单独关闭：

- ``ingest``   把对话切成不可变的 Tier-2 页，并按页生成 Tier-1 摘要单元
- ``query``    默认只查摘要；路由判定不足时才升级到原始日志
- ``commit``   把升级中验证过的事实写回摘要索引

与 ``agents/memory.py`` 的关系：两者共用同一个 ``.range`` 目录下的兄弟目录，
互不覆盖。TierMem 不替代长期记忆文件，而是给它补上「原始日志 + 溯源 + 按需升级」
这三件它现在没有的东西。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from . import config, prompts
from .index import Tier1Index, Unit
from .llm import LLMProtocol, extract_json
from .router import sufficiency_router
from .store import Tier2Store, est_tokens
from .writeback import parse_facts, writeback


def _count_text(text: str) -> tuple[int, int]:
    """返回 ``(CJK 字符数, 其余字符数)``。"""
    cjk = sum(1 for ch in text if config.CJK_START <= ord(ch) <= config.CJK_END)
    return cjk, len(text) - cjk


def _est_from_counts(cjk: int, other: int) -> int:
    """与 ``store.est_tokens`` 完全一致的估算，但用累计计数算，避免重复扫描。

    不能把每行的 ``est_tokens`` 直接相加：``est_tokens`` 对整串做 ``other // 4``，
    逐行相加会因为每行的取整与换行符而系统性偏小，导致最终页超预算。
    """
    return cjk * config.TOKENS_PER_CJK_CHAR + max(1, other // max(1, config.CHARS_PER_TOKEN))


class TierMem:
    """两级记忆的门面。所有模型调用都是 async，可以直接被 Agent 的循环 await。"""

    def __init__(
        self,
        root: Path | str | None = None,
        llm: LLMProtocol | None = None,
        *,
        summary_mode: str | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else config.memory_root()
        self.root.mkdir(parents=True, exist_ok=True)
        if llm is None:
            # 本包不读环境变量：凭据由调用方解析后注入（agents/main.py 负责）
            raise RuntimeError("TierMem 需要显式传入 llm；本包不读环境变量")
        self.llm: LLMProtocol = llm
        self.summary_mode = (summary_mode or config.SUMMARY_MODE).lower()
        if self.summary_mode not in config.SUMMARY_MODES:
            self.summary_mode = "retention"
        self.enabled = config.ENABLED if enabled is None else enabled

        self.store = Tier2Store(self.root)
        self.index = Tier1Index(self.root)

        self._page_vectors: dict[str, list[float]] = {}
        self._vectors_path = self.root / config.TIER2_VECTORS_FILENAME
        self._load_page_vectors()

        # 本会话已注入的字节数，超预算后只允许 ANSWER 路径
        self.session_bytes = 0

    # ── 页向量缓存 ─────────────────────────────────────────
    def _load_page_vectors(self) -> None:
        self._page_vectors = {}
        if not self._vectors_path.exists():
            return
        try:
            import json

            data = json.loads(self._vectors_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return
        if isinstance(data, dict):
            self._page_vectors = {str(key): list(value) for key, value in data.items()}

    def save_page_vectors(self) -> None:
        import json

        self._vectors_path.write_text(
            json.dumps(self._page_vectors), encoding="utf-8"
        )

    # ── 消息归一化与切页 ───────────────────────────────────
    @staticmethod
    def message_text(message: dict[str, Any]) -> str:
        """兼容两种消息形状：``{role, content}`` 与 ``{ts, speaker, text}``。"""
        if isinstance(message.get("text"), str):
            return message["text"]
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(str(block.get("text") or ""))
                    elif block.get("type") == "tool_result":
                        parts.append("TOOL_RESULT\n" + str(block.get("content") or ""))
                    elif block.get("type") == "tool_use":
                        parts.append(f"TOOL_USE {block.get('name', '')} {block.get('input', '')}")
                    elif "content" in block:
                        parts.append(str(block.get("content") or ""))
            return "\n".join(part for part in parts if part)
        return ""

    def chunk_messages(
        self,
        messages: list[dict[str, Any]],
        page_tokens: int | None = None,
    ) -> list[str]:
        """按目标 token 数把消息聚成页。

        单条消息超过页上限时会被切碎（优先按换行，其次按字符），续段标注 ``(n/m)``。
        **页大小必须有界**：原实现不切碎单条消息，于是一条 22KB 的工具结果独占整页，
        摘要只覆盖开头，升级读取时又被 ``MAX_EVIDENCE_BYTES`` 截断到 8KB，
        超出部分就永远取不回来了。
        """
        limit = page_tokens if page_tokens is not None else config.PAGE_TARGET_TOKENS
        limit = max(int(limit), config.MIN_PAGE_TOKENS)
        pages: list[str] = []
        buffer: list[str] = []
        cjk = 0
        other = 0

        def flush() -> None:
            nonlocal buffer, cjk, other
            if buffer:
                pages.append("\n".join(buffer))
            buffer, cjk, other = [], 0, 0

        def push(line: str) -> None:
            nonlocal cjk, other
            line_cjk, line_other = _count_text(line)
            sep = 1 if buffer else 0
            if buffer and _est_from_counts(cjk + line_cjk, other + line_other + sep) > limit:
                flush()
                sep = 0
            buffer.append(line)
            cjk += line_cjk
            other += line_other + sep

        for message in messages:
            if not isinstance(message, dict):
                continue
            text = self.message_text(message)
            if not text:
                continue
            ts = message.get("ts", "")
            speaker = message.get("speaker") or message.get("role") or "user"
            prefix = config.PAGE_LINE_TEMPLATE.format(ts=ts, speaker=speaker, text="")
            parts = self._split_oversized(text, max(1, limit - est_tokens(prefix)))
            total = len(parts)
            for index, part in enumerate(parts, 1):
                body = part if total == 1 else f"({index}/{total}) {part}"
                line = config.PAGE_LINE_TEMPLATE.format(ts=ts, speaker=speaker, text=body)
                if _est_from_counts(*_count_text(line)) > limit:
                    # 前缀与序号标签也可能把单行顶过预算，此时按字符再切一次，
                    # 保证「页大小有界」对任何输入都成立。
                    for piece in self._hard_split_line(line, limit):
                        push(piece)
                else:
                    push(line)
        flush()
        return pages

    @staticmethod
    def _split_oversized(text: str, budget_tokens: int) -> list[str]:
        """把超长文本切成每段不超过 ``budget_tokens`` 的片段。

        优先在换行处切（工具日志天然按行）；单行本身就超预算时按字符硬切。
        """
        if budget_tokens <= 0 or est_tokens(text) <= budget_tokens:
            return [text]
        parts: list[str] = []
        group: list[str] = []
        cjk = 0
        other = 0

        def flush() -> None:
            nonlocal group, cjk, other
            if group:
                parts.append("\n".join(group))
            group, cjk, other = [], 0, 0

        for line in text.splitlines():
            line_cjk, line_other = _count_text(line)
            if _est_from_counts(line_cjk, line_other) > budget_tokens:
                flush()
                parts.extend(TierMem._hard_split_line(line, budget_tokens))
                continue
            sep = 1 if group else 0
            if group and _est_from_counts(cjk + line_cjk, other + line_other + sep) > budget_tokens:
                flush()
                sep = 0
            group.append(line)
            cjk += line_cjk
            other += line_other + sep
        flush()
        return parts or [text]

    @staticmethod
    def _hard_split_line(line: str, budget_tokens: int) -> list[str]:
        """单行超预算时按字符硬切，保证每一片都不超过预算。"""
        parts: list[str] = []
        buf: list[str] = []
        cjk = 0
        other = 0
        for ch in line:
            is_cjk = config.CJK_START <= ord(ch) <= config.CJK_END
            if buf and _est_from_counts(
                cjk + (1 if is_cjk else 0),
                other + (0 if is_cjk else 1),
            ) > budget_tokens:
                parts.append("".join(buf))
                buf, cjk, other = [], 0, 0
            buf.append(ch)
            if is_cjk:
                cjk += 1
            else:
                other += 1
        if buf:
            parts.append("".join(buf))
        return parts

    # ── 阶段 1：摄入 ───────────────────────────────────────
    async def ingest(
        self,
        messages: list[dict[str, Any]],
        session: str = "default",
        page_tokens: int | None = None,
    ) -> dict[str, Any]:
        """把一段会话写入两级存储。返回本次新增的页数与单元数。

        **本方法是追加式的，不做去重：传进来的每一条消息都会变成新页，重复提交
        同一段历史会产生重复页，而不是被识别为「已存在」。调用方必须只提交增量
        —— 见 ``agents/agent.py`` 的 ``_tier_ingest_cursor``（实测漏了这一点时，
        一轮 20 次工具调用会让页数膨胀 10.7 倍）。

        顺序很关键：**先落 Tier-2，再生成摘要**。如果先摘要后落盘，
        摘要一旦生成完就因为进程退出而丢掉原始页，溯源链接就指向了不存在的证据。
        """
        if not self.enabled:
            return {"pages": 0, "units": 0, "skipped": "disabled"}

        pages = self.chunk_messages(messages, page_tokens)
        if not pages:
            return {"pages": 0, "units": self.index.count()}

        page_ids = self.store.append_pages(pages, session=session)

        # 只为新页算向量，命中缓存的不重复调用
        fresh = [pid for pid in page_ids if not self._page_vectors.get(pid)]
        if fresh:
            texts = [str((self.store.get(pid) or {}).get("text") or "") for pid in fresh]
            try:
                vectors = await self.llm.embed(texts)
            except Exception:  # noqa: BLE001
                vectors = [[] for _ in fresh]
            for pid, vector in zip(fresh, vectors):
                self._page_vectors[pid] = list(vector or [])
            self.save_page_vectors()

        system = prompts.summary_system(self.summary_mode)
        for pid in page_ids:
            page = self.store.get(pid) or {}
            raw = str(page.get("text") or "")
            summary = ""
            try:
                summary = (
                    await self.llm.chat(system, raw, max_tokens=config.SUMMARY_MAX_TOKENS)
                ).strip()
            except Exception:  # noqa: BLE001
                summary = ""
            if len(summary) < config.SUMMARY_MIN_CHARS:
                # 摘要失败就退化为截取原文，宁可占用上下文也不能让这页没有单元
                summary = raw[: config.SUMMARY_FALLBACK_CHARS]

            try:
                vectors = await self.llm.embed([summary])
                vector = list(vectors[0]) if vectors else []
            except Exception:  # noqa: BLE001
                vector = []
            # 溯源链接在这里直接写死为这一页，不经过模型
            self.index.add(summary, [pid], vector, origin="ingest")

        return {"pages": len(page_ids), "units": self.index.count()}

    # ── 检索 ───────────────────────────────────────────────
    async def retrieve_tier1(self, query: str, k: int | None = None) -> list[Unit]:
        top_k = k if k is not None else config.TIER1_TOP_K
        if self.index.count() == 0:
            return []
        try:
            vectors = await self.llm.embed([query])
            vector = list(vectors[0]) if vectors else []
        except Exception:  # noqa: BLE001
            vector = []
        if not vector:
            # 没有向量能力时退化为按写入顺序取最新的若干条
            return self.index.all()[-top_k:]
        return [unit for unit, _ in self.index.search(vector, top_k)]

    async def retrieve_tier2(
        self,
        query: str,
        k: int | None = None,
        exclude: set[str] | None = None,
    ) -> list[dict]:
        top_k = k if k is not None else config.TIER2_TOP_K
        pages = self.store.all()
        if not pages:
            return []
        try:
            vectors = await self.llm.embed([query])
            vector = list(vectors[0]) if vectors else []
        except Exception:  # noqa: BLE001
            vector = []
        if not vector:
            return [page for page in pages if not (exclude and page["page_id"] in exclude)][:top_k]

        from .index import cosine

        scored: list[tuple[dict, float]] = []
        for page in pages:
            if exclude and page["page_id"] in exclude:
                continue
            page_vector = self._page_vectors.get(page["page_id"])
            if not page_vector:
                continue
            scored.append((page, cosine(vector, page_vector)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [page for page, _ in scored[:top_k]]

    def _linked_pages(self, units: list[Unit]) -> list[dict]:
        """把若干单元的溯源链接展开成原始页，按链接出现顺序去重。"""
        page_ids: list[str] = []
        for unit in units:
            page_ids.extend(unit.provenance)
        return self.store.many(page_ids)

    @staticmethod
    def _clip(text: str) -> str:
        raw = str(text or "")
        if len(raw.encode()) <= config.MAX_EVIDENCE_BYTES:
            return raw
        # 按字节截断要落回字符边界，否则多字节字符会被切坏
        budget = config.MAX_EVIDENCE_BYTES
        clipped = raw.encode()[:budget].decode("utf-8", errors="ignore")
        return clipped + "\n[... 已截断 ...]"

    async def _answer(self, question: str, evidence: list[tuple[str, str]]) -> str:
        if not evidence:
            return "（TierMem 没有检索到任何证据）"
        block = "\n\n".join(f"--- {label} ---\n{self._clip(text)}" for label, text in evidence)
        try:
            return (
                await self.llm.chat(
                    prompts.ANSWER_SYSTEM,
                    f"Evidence:\n{block}\n\nQuestion: {question}",
                    max_tokens=config.ANSWER_MAX_TOKENS,
                )
            ).strip()
        except Exception as exc:  # noqa: BLE001
            return f"（回答生成失败：{exc}）"

    # ── 阶段 2：查询 ───────────────────────────────────────
    async def query(
        self,
        question: str,
        k: int | None = None,
        page_budget: int | None = None,
    ) -> dict[str, Any]:
        """回答一个问题，返回带完整轨迹的 trace，便于审计与评测。"""
        started = time.time()
        top_k = k if k is not None else config.TIER1_TOP_K
        budget = page_budget if page_budget is not None else config.escalation_page_budget()

        trace: dict[str, Any] = {
            "question": question,
            "route": None,
            "steps": [],
            "summary_mode": self.summary_mode,
        }

        if not self.enabled:
            trace.update(mode="disabled", answer="（TierMem 已关闭）", seconds=0.0)
            return trace

        # 快路径：先只看摘要
        units = await self.retrieve_tier1(question, top_k)
        trace["retrieved_units"] = [unit.unit_id for unit in units]

        if not units:
            # 索引为空时任何路由判定都没有意义：既没有摘要可读，也没有页可升级。
            # 这里直接返回，避免白花掉「路由 + 回答」两次模型调用。
            trace["mode"] = "empty"
            trace["route"] = {
                "action": config.ROUTE_ESCALATE,
                "missing": ["empty index"],
                "reason": "no unit indexed yet",
            }
            trace["answer"] = ""
            trace["seconds"] = round(time.time() - started, 3)
            return trace

        route = await sufficiency_router(
            self.llm, question, [(unit.unit_id, unit.text) for unit in units]
        )
        trace["route"] = route

        over_budget = self.session_bytes >= config.MAX_SESSION_BYTES
        can_escalate = (
            route["action"] == config.ROUTE_ESCALATE
            and config.ENABLE_ESCALATION
            and config.MAX_ESCALATION_HOPS > 0
            and budget > 0
            and not over_budget
        )
        if not can_escalate:
            trace["mode"] = "answer"
            if over_budget:
                trace["budget_exhausted"] = True
            evidence = [(f"摘要 {unit.unit_id}", unit.text) for unit in units]
            trace["steps"].append({"tier": "tier1", "units": [unit.unit_id for unit in units]})
            trace["answer"] = await self._answer(question, evidence)
            self._charge(evidence)
            trace["seconds"] = round(time.time() - started, 3)
            return trace

        # 升级：先读被召回摘要显式链接的页（溯源引导）
        seeds = self._linked_pages(units)[:budget]
        evidence: list[tuple[str, str]] = [(f"摘要 {unit.unit_id}", unit.text) for unit in units]
        evidence.extend((f"原始页 {page['page_id']}", page["text"]) for page in seeds)
        trace["steps"].append({
            "tier": "tier2",
            "seed_pages": [page["page_id"] for page in seeds],
            "desc": "provenance-linked pages",
        })

        seen = {page["page_id"] for page in seeds}
        sub_query = question
        for hop in range(config.MAX_ESCALATION_HOPS):
            remaining = budget - len(seen)
            if remaining <= 0:
                break
            extra = await self.retrieve_tier2(sub_query, min(config.TIER2_TOP_K, remaining), exclude=seen)
            if not extra:
                break
            seen.update(page["page_id"] for page in extra)
            evidence.extend((f"原始页 {page['page_id']}", page["text"]) for page in extra)
            trace["steps"].append({
                "tier": "tier2",
                "hop": hop + 1,
                "pages": [page["page_id"] for page in extra],
            })
            if len(seen) >= budget:
                break

            preview = "\n".join(self._clip(text)[: config.PLAN_EVIDENCE_PREVIEW_CHARS] for _, text in evidence)
            try:
                plan_reply = await self.llm.chat(
                    prompts.PLAN_SYSTEM,
                    f"Question: {question}\n\nEvidence so far:\n{preview}",
                    max_tokens=config.PLAN_MAX_TOKENS,
                )
                plan = extract_json(plan_reply)
            except Exception:  # noqa: BLE001
                break
            if plan.get("stop") or not plan.get("query"):
                break
            sub_query = str(plan["query"])

        trace["mode"] = "escalated"
        trace["pages_read"] = sorted(seen)
        trace["answer"] = await self._answer(question, evidence)
        self._charge(evidence)
        trace["seconds"] = round(time.time() - started, 3)
        return trace

    def _charge(self, evidence: list[tuple[str, str]]) -> None:
        """把本次注入的证据计入会话预算。"""
        self.session_bytes += sum(len(self._clip(text).encode()) for _, text in evidence)

    def reset_session_budget(self) -> None:
        """归零注入预算。

        ``session_bytes`` 统计的是「本会话已经往上下文里塞了多少 TierMem 证据」。
        历史一旦被归档替换（``agents/agent.py::_archive_messages``），这些注入就
        随历史一起消失了，计数必须同步归零。否则它只增不减，耗尽后
        ``over_budget`` 会让升级在整个进程生命周期内永久失效。
        """
        self.session_bytes = 0

    # ── 阶段 3：写回 ───────────────────────────────────────
    async def commit(
        self,
        question: str,
        trace: dict[str, Any],
        max_facts: int | None = None,
    ) -> list[dict[str, Any]]:
        """把本次升级读到的结论沉淀进摘要索引。

        只提交与触发查询相关度最高的若干条——升级过程往往读到几十条原子事实，
        全量写回会让索引迅速膨胀且大量重复。
        """
        if trace.get("mode") != "escalated" or not config.ENABLE_WRITEBACK:
            return []

        page_ids = [
            pid
            for step in trace.get("steps", [])
            for pid in (step.get("seed_pages") or step.get("pages") or [])
        ]
        pages = self.store.many(page_ids)
        if not pages:
            return []

        raw = "\n\n".join(str(page.get("text") or "") for page in pages)
        try:
            reply = await self.llm.chat(
                prompts.FACT_EXTRACT_SYSTEM,
                f"Question that triggered escalation: {question}\n\nRaw evidence:\n{raw}",
                max_tokens=config.FACT_EXTRACT_MAX_TOKENS,
            )
        except Exception as exc:  # noqa: BLE001
            return [{"op": "REJECT", "reason": f"fact extraction failed: {exc}"}]

        facts = parse_facts(reply)
        if not facts:
            # 多半是被 max_tokens 截断导致 JSON 不完整，把回复开头留下来便于定位
            return [{"op": "NO_FACT_PARSED", "reply_head": (reply or "")[:300]}]

        limit = max_facts if max_facts is not None else config.WRITEBACK_MAX_FACTS
        if limit > 0 and len(facts) > limit:
            try:
                # 按与触发查询的相似度排序，只留最相关的那几条
                from .index import cosine

                vectors = await self.llm.embed([question] + [item["fact"] for item in facts])
                if vectors and vectors[0]:
                    question_vector = vectors[0]
                    ranked = sorted(
                        zip(facts, vectors[1:]),
                        key=lambda pair: cosine(question_vector, pair[1]),
                        reverse=True,
                    )
                    facts = [item for item, _ in ranked[:limit]]
                else:
                    facts = facts[:limit]
            except Exception:  # noqa: BLE001
                facts = facts[:limit]

        return await writeback(self.index, self.llm, facts, pages)

    # ── 供 Agent 接入 ──────────────────────────────────────
    @staticmethod
    def format_for_injection(trace: dict[str, Any]) -> str:
        """把一次查询结果包成可注入消息的块，格式与 ``agents/memory.py`` 一致。"""
        if not trace.get("answer"):
            return ""
        route = trace.get("route") or {}
        header = f"TierMem（{trace.get('mode')}，路由={route.get('action')}）"
        if trace.get("pages_read"):
            header += f"，读取原始页={trace['pages_read']}"
        return (
            "<system-reminder>\n"
            f"{header}\n\n{trace['answer']}\n"
            "</system-reminder>"
        )

    def build_prompt_section(self, max_units: int = 50) -> str:
        """生成注入 system prompt 的说明与索引摘要。"""
        units = self.index.all()
        lines = [
            "# TierMem（两级记忆）",
            "",
            f"数据目录：`{self.root}`",
            "- Tier-2 原始日志：不可变分页，只追加，页号稳定，位于 "
            f"`{config.TIER2_PAGES_FILENAME}`",
            "- Tier-1 摘要索引：每条摘要都带指向页号的溯源链接，位于 "
            f"`{config.TIER1_INDEX_FILENAME}`",
            "- 默认只读摘要；摘要缺少决定性约束（否定、精确值、归因、顺序）时才升级读原始页。",
            "",
            f"当前规模：{self.store.count()} 页 / {self.index.count()} 个单元"
            f"（摘要模式：{self.summary_mode}）",
        ]
        index_block = self.render_index(max_units)
        if index_block:
            lines.append("")
            lines.append(index_block)
        return "\n".join(lines)

    def render_index(self, max_units: int = 60) -> str:
        """把摘要索引渲染成可读片段，供归档说明与 system prompt 复用。"""
        units = self.index.all()
        if not units:
            return ""
        lines = ["## Tier-1 摘要索引"]
        for unit in units[-max_units:]:
            preview = unit.text.replace("\n", " ")[:80]
            lines.append(f"- `{unit.unit_id}` [{unit.origin}] 溯源={unit.provenance} — {preview}")
        return "\n".join(lines)

    def stats(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "pages": self.store.count(),
            "units": self.index.count(),
            "summary_mode": self.summary_mode,
            "session_bytes": self.session_bytes,
            "enabled": self.enabled,
        }
