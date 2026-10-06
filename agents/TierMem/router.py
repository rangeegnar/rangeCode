"""运行时充分性路由：决定「只用摘要回答」还是「升级去读原始日志」。

论文最强的一条失败模式是**假缓存命中**——摘要和问题共享关键词，路由就以为够了，
实际缺少决定性约束。所以这里的提示词强制逐项检查否定、精确值、归因、范围与顺序，
并且明确要求「共享关键词不等于充分」。

路由不可用或解析失败时的默认动作由 ``config.DEFAULT_ROUTE_ON_ERROR`` 决定，
默认是 ESCALATE：拿不准就去查原文，宁可多花 token 也不要给出无法追溯的答案。
"""

from __future__ import annotations

from . import config, prompts
from .llm import LLMProtocol, extract_json


def _fallback(reason: str) -> dict:
    return {
        "action": config.DEFAULT_ROUTE_ON_ERROR,
        "missing": [reason],
        "reason": reason,
    }


async def sufficiency_router(
    llm: LLMProtocol,
    query: str,
    summaries: list[tuple[str, str]],
) -> dict:
    """判断摘要是否足以回答。

    ``summaries`` 是 ``(unit_id, text)`` 列表——只给摘要正文，不给原始页，
    这样路由的判定成本与摘要数量成正比，而与原始日志无关。
    """
    if not config.ENABLE_ROUTER:
        return {"action": config.ROUTE_ANSWER, "missing": [], "reason": "router disabled"}

    if not summaries:
        if config.ESCALATE_ON_EMPTY_INDEX:
            return _fallback("no summary retrieved")
        return {"action": config.ROUTE_ANSWER, "missing": [], "reason": "empty index, answer path"}

    block = "\n".join(f"[{unit_id}] {text}" for unit_id, text in summaries)
    try:
        reply = await llm.chat(
            prompts.ROUTER_SYSTEM,
            f"Query: {query}\n\nRetrieved summaries:\n{block}",
            max_tokens=config.ROUTER_MAX_TOKENS,
        )
    except Exception as exc:  # noqa: BLE001 - 路由失败绝不能拖垮主对话
        return _fallback(f"router call failed: {exc}")

    try:
        data = extract_json(reply)
    except Exception:  # noqa: BLE001
        return {"action": config.DEFAULT_ROUTE_ON_ERROR, "missing": ["router output unparsable"],
                "reason": (reply or "")[:200]}

    action = str(data.get("action") or "").upper()
    if action not in config.ROUTE_ACTIONS:
        action = config.DEFAULT_ROUTE_ON_ERROR
    return {
        "action": action,
        "missing": list(data.get("missing") or []),
        "reason": str(data.get("reason") or "")[:300],
    }


async def judge(llm: LLMProtocol, query: str, answer: str, reference: str) -> dict:
    """评测用判官：判对错，并单独标注「是否因为证据里缺了决定性约束而错」。

    第二个指标就是论文的 Unverifiable Omission Rate，用来衡量写时压缩造成的损失。
    """
    try:
        reply = await llm.chat(
            prompts.JUDGE_SYSTEM,
            f"Query: {query}\n\nReference answer: {reference}\n\nAssistant answer: {answer}",
            max_tokens=config.JUDGE_MAX_TOKENS,
        )
        data = extract_json(reply)
    except Exception:  # noqa: BLE001
        return {"correct": False, "unverifiable_omission": False, "why": "judge unparsable"}
    return {
        "correct": bool(data.get("correct")),
        "unverifiable_omission": bool(data.get("unverifiable_omission")),
        "why": str(data.get("why") or "")[:300],
    }
