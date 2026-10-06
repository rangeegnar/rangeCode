"""TierMem 用到的全部提示词模板。

刻意集中在一个文件：提示词是调参的主要对象，散落在各模块里会很难对照实验。
需要按环境切换的部分（词数上限、检索模式）从 ``config`` 插值，不写字面量。
"""

from __future__ import annotations

from . import config

# ─────────────────────────── 写入侧：按页生成摘要 ───────────────────────────

SUMMARY_SYSTEM_LOSSY = f"""Summarize one segment of an engineering conversation.

Capture the topic, the direction of the work, and the decisions taken.
Write at most {config.SUMMARY_WORD_CAP_LOSSY} words. Output the summary text only."""

SUMMARY_SYSTEM_RETENTION = f"""Summarize one segment of an engineering conversation.

Capture the topic and the direction of the work, and you must preserve every
exact value, negation or prohibition, attribution, and ordering constraint
that appears in the segment. Never replace a number with a rounded value.
Never drop a "must not" or "does not need" statement.
Write at most {config.SUMMARY_WORD_CAP_RETENTION} words. Output the summary text only."""


def summary_system(mode: str | None = None) -> str:
    """按模式取摘要提示词。未知模式一律用 retention（保住约束优先）。"""
    resolved = (mode or config.SUMMARY_MODE).lower()
    if resolved == "lossy":
        return SUMMARY_SYSTEM_LOSSY
    return SUMMARY_SYSTEM_RETENTION


# ─────────────────────────── 路由：充分性判定 ───────────────────────────

ROUTER_SYSTEM = f"""You are the sufficiency router of a two-tier memory system.

You receive a user query and the summaries retrieved from the summary index.
Decide whether those summaries contain enough evidence to answer the query
faithfully, or whether the raw conversation log must be consulted.

Escalate when the summary lacks any of the following that the query depends on:
- a negation, prohibition or exclusion
- an exact value: number, threshold, version, path, identifier, date
- attribution: who required or decided something
- scope or ordering constraints, required sequence of steps
- any condition that changes the answer

Do not escalate merely because the summary shares keywords with the query.
Topical similarity is not sufficiency. If the summary states the theme but not
the decisive constraint, escalate.

Return only one JSON object:
{{"action": "{config.ROUTE_ANSWER}" | "{config.ROUTE_ESCALATE}", "missing": ["what is missing, empty if none"], "reason": "one sentence"}}"""


# ─────────────────────────── 回答 ───────────────────────────

ANSWER_SYSTEM = f"""You answer a question using only the supplied evidence.

Requirements:
- Base every claim on the evidence. Do not add facts that are not present.
- If the evidence does not contain a decisive constraint the question depends on,
  say explicitly that the available evidence is insufficient and name what is missing.
- Cite which evidence item supports each claim.
- Answer in the same language as the question.
- Keep the answer under roughly {config.ANSWER_MAX_TOKENS} tokens."""


# ─────────────────────────── 升级：多跳检索规划 ───────────────────────────

PLAN_SYSTEM = """You decide whether another search over the raw log is needed before answering.

Return only one JSON object: {"stop": true|false, "query": "next search query"}

Set stop=true when the evidence already covers the decision the question depends on,
or when another search is unlikely to add anything."""


# ─────────────────────────── 写回：事实抽取 ───────────────────────────

FACT_EXTRACT_SYSTEM = """Extract atomic facts from the raw evidence below.

Each fact must be self-contained and must preserve negations, exact values,
attribution and ordering. For every fact also return the verbatim quote from
the evidence that supports it. Do not invent page identifiers.
Do not output more than the most important facts; keep the JSON complete.

Return only one JSON object:
{"facts": [{"fact": "...", "evidence_quote": "..."}]}"""


# ─────────────────────────── 写回：冲突裁决 ───────────────────────────

MANAGER_SYSTEM = f"""You maintain a summary memory index.

You are given a new fact verified against the raw log, and the most similar
existing summary units. Choose exactly one operation:

- "{config.WB_SKIP}": the new fact is already entailed by an existing unit, nothing to add.
- "{config.WB_UPDATE}": the new fact adds precision to an existing unit (a concrete value,
  name, date, negation or condition). Produce the merged text. Also return
  "target_unit_id" with the existing unit id you are updating.
- "{config.WB_ADD}": the new fact is an independent event or constraint not covered.

Never discard a negation, an exact value, an attribution or an ordering
constraint that appears in the new fact.

Return only one JSON object:
{{"op": "{config.WB_SKIP}"|"{config.WB_UPDATE}"|"{config.WB_ADD}", "target_unit_id": "existing unit id or empty", "merged": "merged text, empty unless UPDATE", "text": "text to store for ADD"}}

For UPDATE provenance is inherited automatically; do not output provenance ids."""


# ─────────────────────────── 评测：判官 ───────────────────────────

JUDGE_SYSTEM = """You grade an assistant answer against a reference answer.

Return only one JSON object:
{"correct": true|false, "unverifiable_omission": true|false, "why": "one sentence"}

correct = the answer is faithful and complete with respect to the reference.
unverifiable_omission = the answer is wrong specifically because a decisive
constraint (negation, exact value, attribution, ordering) was absent from the
evidence the assistant was given. Set it false for other error causes."""
