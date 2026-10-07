"""运行入口：跑图、采集证据、检查不变量。

**不落盘。** 运行记录由 `records.py` 负责，调用方（`interface.execute_multiagent`）
决定写到哪个会话目录。
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from .config import DEFAULT_BUDGET, Budget
from .graph import make_graph
from .llm import model_config
from .scheduling import check_invariants, summarise
from .state import initial_state


def parallelism_evidence(events: list[dict[str, Any]]) -> dict[str, Any]:
    """用区间重叠判断 worker 是否真并行。

    串行时"单次耗时之和" ≈ "区间并集长度"；并行时并集明显小于总和。
    重叠系数 = 和 / 并集，> 1 即为并发。
    """
    spans = [e for e in events if e.get("node") == "worker" and "start" in e]
    if not spans:
        return {"spans": 0, "sum_seconds": 0.0, "union_seconds": 0.0, "overlap_factor": 1.0}

    ordered = sorted(spans, key=lambda s: s["start"])
    union = 0.0
    cur_start, cur_end = ordered[0]["start"], ordered[0]["end"]
    for s in ordered[1:]:
        if s["start"] > cur_end:
            union += cur_end - cur_start
            cur_start, cur_end = s["start"], s["end"]
        else:
            cur_end = max(cur_end, s["end"])
    union += cur_end - cur_start

    total = sum(s["seconds"] for s in spans)
    return {
        "spans": len(spans),
        "sum_seconds": round(total, 2),
        "union_seconds": round(union, 2),
        "overlap_factor": round(total / union, 2) if union > 0 else 1.0,
    }


def cost_by_node(events: list[dict[str, Any]]) -> dict[str, int]:
    """按节点归集 token。对标 VMAO 的「execution 占 61%」。"""
    out: dict[str, int] = {}
    for e in events:
        node = str(e.get("node"))
        out[node] = out.get(node, 0) + int(e.get("tokens") or 0)
    return out


def cost_by_model(events: list[dict[str, Any]]) -> dict[str, int]:
    """按模型归集 token。执行与验证是两个模型，要分开看。

    没有 model 字段的事件（如 join，它是纯路由节点不计 token）直接跳过，
    避免出现一个无意义的 "?" 零值项。
    """
    out: dict[str, int] = {}
    for e in events:
        model = e.get("model")
        tokens = int(e.get("tokens") or 0)
        if not model or tokens == 0:
            continue
        out[str(model)] = out.get(str(model), 0) + tokens
    return out


async def run_orchestration(query: str, *, budget: Budget = DEFAULT_BUDGET) -> dict[str, Any]:
    """跑一次完整编排，返回可核对的运行记录。

    **不负责落盘。** 记录写到哪由调用方决定（见 `records.py` 与 `interface.py`），
    因为目录位置与生命周期是另一个变化原因。
    """
    cfg = model_config()
    graph = make_graph(budget)

    started = time.monotonic()
    final = await graph.ainvoke(
        initial_state(query, exec_model=cfg.exec_model, verify_model=cfg.verify_model,
                      verify_is_independent=cfg.verify_is_independent),
        config={"recursion_limit": budget.graph_recursion_limit})
    elapsed = time.monotonic() - started

    violations = check_invariants(final, budget)
    events = final.get("events", [])

    # 断言级检查：verify/replan 必须真的走了验证模型
    verify_events = [e for e in events if e.get("node") == "verify"]
    verify_wrong_model = [e for e in verify_events if e.get("model") != cfg.verify_model]
    if verify_wrong_model:
        violations.append(
            f"I6 违反：验证节点使用的模型不是 {cfg.verify_model}，"
            f"实际为 {[e.get('model') for e in verify_wrong_model]}")

    record: dict[str, Any] = {
        "query": query,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_seconds": round(elapsed, 2),
        "models": {
            "exec": cfg.exec_model,
            "verify": cfg.verify_model,
            "verify_is_independent": cfg.verify_is_independent,
        },
        "summary": summarise(final),
        "invariant_violations": violations,
        "parallelism": parallelism_evidence(events),
        "cost_by_node": cost_by_node(events),
        "cost_by_model": cost_by_model(events),
        "tasks": final.get("tasks", []),
        "results": final.get("results", []),
        "verdicts": final.get("verdicts", []),
        "intercepts": final.get("intercepts", []),
        "events": events,
        "plan_warnings": final.get("plan_warnings", []),
        "final_answer": final.get("final_answer", ""),
        "stop_reason": final.get("stop_reason", ""),
    }
    return record
