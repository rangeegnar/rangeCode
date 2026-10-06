"""验证式写回：把升级中读到的结论沉淀回摘要索引。

两条设计要点：

1. **溯源标识符不由模型生成。** 模型只输出 ``(fact, evidence_quote)``，
   页号由 ``deterministic_link`` 用字符串匹配算出来：先对引文做归一化子串匹配，
   失败再退到关键词重叠。这样不会出现模型编造来源的情况。

2. **先冲突检索再裁决。** 每条新事实先召回最相似的若干现有单元，
   再让模型在 SKIP / UPDATE / ADD 里选一个。UPDATE 时新单元继承双方的溯源链接，
   审计链不断。
"""

from __future__ import annotations

import re

from . import config, prompts
from .index import Tier1Index
from .llm import LLMProtocol, extract_json

# 用 [^\w] 切分。注意 Python 的 \w 已经包含 CJK，所以中文连续片段**不会**被切开，
# 需要在下面用 n-gram 自己补，否则中文的关键词回退永远命中不了。
_SPLIT_RE = re.compile(r"[^\w]+", re.UNICODE)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).lower()


def _has_cjk(text: str) -> bool:
    return any(config.CJK_START <= ord(ch) <= config.CJK_END for ch in text)


def tokens_of(text: str) -> list[str]:
    """把一句话切成可比较的 token。

    非 CJK 片段按非字母数字切分后取长度达标的；
    CJK 片段切成 ``CJK_NGRAM`` 长度的 n-gram——这是为了让「引文被改写」
    的情况仍能通过重叠词命中，否则中文事实只能依赖逐字引文。
    """
    out: list[str] = []
    for piece in _SPLIT_RE.split(str(text or "")):
        if not piece:
            continue
        if _has_cjk(piece):
            size = max(1, config.CJK_NGRAM)
            if len(piece) <= size:
                out.append(piece)
            else:
                out.extend(piece[i : i + size] for i in range(len(piece) - size + 1))
        elif len(piece) >= config.MIN_TOKEN_LEN:
            out.append(piece.lower())
    return out


def deterministic_link(fact: str, quote: str, pages: list[dict]) -> tuple[list[str], list[str]]:
    """把一条事实映射回原始页号，返回 ``(page_ids, snippets)``。

    第一步：引文归一化后做子串匹配（精确、可解释）。
    第二步：引文被改写或缺失时，退到关键词重叠（召回优先）。
    两步都没命中就返回空——调用方应据此拒绝这条事实，而不是塞个假来源。
    """
    norm_quote = _normalise(quote)
    hits: list[str] = []
    snippets: list[str] = []
    if norm_quote:
        for page in pages:
            if norm_quote in _normalise(page.get("text")):
                hits.append(page["page_id"])
                snippets.append(str(quote)[: config.EVIDENCE_QUOTE_CLIP])
    if hits:
        return hits, snippets

    if not config.ENABLE_KEYWORD_FALLBACK:
        return [], []

    tokens = tokens_of(fact)
    if not tokens:
        return [], []

    best_page: str | None = None
    best_overlap = 0
    for page in pages:
        norm_text = _normalise(page.get("text"))
        overlap = sum(1 for token in tokens if token in norm_text)
        if overlap > best_overlap:
            best_overlap = overlap
            best_page = page["page_id"]
    if best_page is not None and best_overlap >= config.MIN_KEYWORD_OVERLAP:
        return [best_page], [str(fact)[: config.EVIDENCE_QUOTE_CLIP]]
    return [], []


def parse_facts(reply: str) -> list[dict]:
    """解析事实抽取结果，容忍模型实际会返回的几种形状：
    顶层直接是数组、引文换了 key、以及数组里是纯字符串。"""
    try:
        data = extract_json(reply)
    except Exception:  # noqa: BLE001
        return []
    rows = data.get("facts") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return []
    out: list[dict] = []
    for row in rows:
        if isinstance(row, str):
            row = {"fact": row}
        if not isinstance(row, dict):
            continue
        fact = str(row.get("fact") or row.get("text") or row.get("finding") or "").strip()
        if not fact:
            continue
        quote = row.get("evidence_quote") or row.get("quote") or row.get("evidence") or ""
        out.append({"fact": fact, "evidence_quote": str(quote)})
    return out


async def writeback(
    index: Tier1Index,
    llm: LLMProtocol,
    facts: list[dict],
    pages: list[dict],
    top_k: int | None = None,
) -> list[dict]:
    """提交已验证事实，返回逐条审计日志。"""
    if not config.ENABLE_WRITEBACK:
        return [{"op": "DISABLED", "reason": "writeback disabled"}]

    conflict_k = top_k if top_k is not None else config.WRITEBACK_CONFLICT_TOP_K
    log: list[dict] = []

    for item in facts:
        fact = str(item.get("fact") or "").strip()
        if not fact:
            continue

        sources, _ = deterministic_link(fact, item.get("evidence_quote", ""), pages)
        if not sources:
            # 拿不到溯源就拒绝，绝不让「无源事实」进索引
            log.append({"fact": fact, "op": "REJECT", "reason": "no provenance link"})
            continue

        # 冲突检索：先把最相似的现有单元拿出来，作为裁决上下文
        try:
            vectors = await llm.embed([fact])
            retrieved = index.search(vectors[0], conflict_k) if vectors and vectors[0] else []
        except Exception:  # noqa: BLE001
            retrieved = []
        context = "\n".join(f"[{unit.unit_id}] {unit.text}" for unit, _ in retrieved) or "(index empty)"

        try:
            reply = await llm.chat(
                prompts.MANAGER_SYSTEM,
                f"New verified fact: {fact}\n\nSimilar existing units:\n{context}",
                max_tokens=config.MANAGER_MAX_TOKENS,
            )
        except Exception as exc:  # noqa: BLE001
            log.append({"fact": fact, "op": "REJECT", "reason": f"manager call failed: {exc}",
                        "provenance": sources})
            continue

        try:
            decision = extract_json(reply)
        except Exception:  # noqa: BLE001
            decision = {"op": config.WRITEBACK_FALLBACK_ACTION}

        op = str(decision.get("op") or "").upper()
        if op not in config.WB_ACTIONS:
            op = config.WRITEBACK_FALLBACK_ACTION

        if op == config.WB_SKIP:
            log.append({"fact": fact, "op": config.WB_SKIP, "provenance": sources})
            continue

        if op == config.WB_UPDATE:
            target = str(decision.get("target_unit_id") or "")
            merged = str(decision.get("merged") or fact)
            vectors = await llm.embed([merged])
            unit = index.update(target, merged, sources, vectors[0] if vectors else [])
            if unit is not None:
                log.append({"fact": fact, "op": config.WB_UPDATE, "target": target,
                            "provenance": sources, "unit_id": unit.unit_id})
                continue
            if not config.WRITEBACK_ADD_ON_MISSING_TARGET:
                log.append({"fact": fact, "op": "REJECT", "reason": "update target missing",
                            "provenance": sources})
                continue
            # 找不到目标就退化成 ADD，并在日志里标出来，方便统计退化率
            text = str(decision.get("text") or fact)
            vectors = await llm.embed([text])
            unit = index.add(text, sources, vectors[0] if vectors else [], origin="writeback")
            log.append({"fact": fact, "op": "ADD_FALLBACK", "target": target,
                        "provenance": sources, "unit_id": unit.unit_id})
            continue

        text = str(decision.get("text") or fact)
        vectors = await llm.embed([text])
        unit = index.add(text, sources, vectors[0] if vectors else [], origin="writeback")
        log.append({"fact": fact, "op": config.WB_ADD, "provenance": sources,
                    "unit_id": unit.unit_id})

    return log
