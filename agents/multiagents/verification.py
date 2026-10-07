"""验证与重规划 —— 走**验证模型**（MODEL4TEST / ecnu-plus）。

执行与验证分模型是 VMAO 的核心设计。理由是 MAST 的 14 条失败模式里
"错误验证"占 9.1%，比"无验证"（8.2%）更高频：问题不只是有没有检查，
而是检查得对不对。同一模型既执行又验证，它的系统性盲区会在两个环节同时出现。

另外要求验证器**给高分必须交出可反驳的理由** —— 写不出理由就不许打高分。
"""
from __future__ import annotations

import json

from .config import Budget
from .llm import SLOT_VERIFY, OrchestratorError, call_json
from .state import OrchState, SubResult, SubTask, Verdict

VERIFIER_SYSTEM = """你是一个独立验证者。你的任务不是复述结论，而是找出结论的不足之处。

对每个子问题，按以下判据判断它是否被真正回答：
1. 完整性：子问题的每个方面都被覆盖了吗？部分覆盖只能给 0.3-0.7。
2. 证据质量：结论有没有具体出处（文件路径、行号、工具名与参数）？没有出处的结论不能算完成。
3. 具体性：是具体事实和数字，还是"可能存在"、"通常"这类模糊表述？
4. 矛盾：结果内部、或不同子问题的结果之间有没有互相冲突的说法？
   有矛盾必须写进 contradictions，哪怕你不确定哪边对。
5. 缺失项：明确列出漏掉的部分。没有就留空数组。

打分纪律：你的默认倾向必须是怀疑，不是认可。**给高分的唯一理由是证据充分且覆盖完整**，
reason 里必须指出具体是哪些证据支撑了哪条判据。写不出理由就不许打高分。
只有当结果明确说明自己"查不到 / 不属于本领域"并给出理由时，才接受它是诚实失败。

只输出 JSON 对象：
{{
  "verdicts": [
    {{
      "task_id": "子问题 id",
      "status": "complete 或 partial 或 incomplete",
      "completeness": 0.0到1.0之间的小数,
      "missing_aspects": ["缺失的部分"],
      "contradictions": ["发现的矛盾"],
      "recommendation": "accept 或 retry 或 escalate",
      "reason": "一句话说明打分理由，并指出支撑判据的具体证据"
    }}
  ]
}}
"""

REPLANNER_SYSTEM = """你是一个重规划者。根据验证结论决定下一步动作。

决策逻辑：
- 整体完成度达到 {ready_threshold} 时，返回空列表，表示可以进入综合。
- 只要有 incomplete 的子问题，必须把它们全部放进 retry 列表 —— 这是硬性要求，
  不允许挑着重试，漏掉任何一个都会让最终答案带着已知缺口交付。
- 完成度在 0.5 到 {ready_threshold} 之间时，可以补少量新子问题去填补缺失方面。
- 发现矛盾时，补充针对不同证据来源的子问题去求证。
- 迭代已达 {iteration}/{max_iterations} 时，返回空列表停止。

只输出 JSON 对象：
{{
  "retry_sub_questions": ["需要重试的子问题 id"],
  "new_sub_questions": ["需要新增的子问题 id，通常为空"],
  "explanation": "为什么这样决定"
}}
"""


def _coerce_verdict(obj: dict, valid_ids: set[str]) -> Verdict | None:
    tid = str(obj.get("task_id") or "").strip()
    if tid not in valid_ids:
        return None
    status = str(obj.get("status") or "").strip().lower()
    if status not in {"complete", "partial", "incomplete"}:
        status = "partial"
    rec = str(obj.get("recommendation") or "").strip().lower()
    if rec not in {"accept", "retry", "escalate"}:
        rec = "accept" if status == "complete" else "retry"
    try:
        score = float(obj.get("completeness", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    return Verdict(
        task_id=tid,
        status=status,
        completeness=max(0.0, min(1.0, score)),
        missing_aspects=[str(x) for x in (obj.get("missing_aspects") or [])][:10],
        contradictions=[str(x) for x in (obj.get("contradictions") or [])][:10],
        recommendation=rec,
        reason=str(obj.get("reason") or "")[:600],
    )


def _render_results(tasks: list[SubTask], results: list[SubResult]) -> str:
    by_id = {t["id"]: t for t in tasks}
    blocks = []
    for r in results:
        task = by_id.get(r["task_id"], {})
        blocks.append(json.dumps({
            "task_id": r["task_id"],
            "domain": r["domain"],
            "engine": r["engine"],
            "question": task.get("question", ""),
            "verification_criteria": task.get("verification_criteria", ""),
            "result_ok": r["ok"],
            "result_text": r["text"][:4000],
            "sources": r["sources"],
            "error": r["error"],
            "observation_purified": r.get("purified", False),
        }, ensure_ascii=False))
    return "\n".join(blocks)


def verify(query: str, tasks: list[SubTask], results: list[SubResult]) -> tuple[list[Verdict], int]:
    """对一批结果打分。返回 (verdicts, token)。走验证模型。"""
    if not results:
        return [], 0
    ids = {t["id"] for t in tasks}
    payload, tokens = call_json(
        VERIFIER_SYSTEM,
        f"# 原始任务\n{query}\n\n# 待验证的子问题与结果\n{_render_results(tasks, results)}",
        slot=SLOT_VERIFY,
    )
    if not isinstance(payload, dict):
        raise OrchestratorError("验证器返回的不是 JSON 对象")
    raw = payload.get("verdicts")
    if not isinstance(raw, list):
        raise OrchestratorError(
            f"验证器输出里没有 verdicts 数组：{json.dumps(payload, ensure_ascii=False)[:300]}")

    verdicts = [v for v in (_coerce_verdict(o, ids) for o in raw if isinstance(o, dict)) if v]

    # 验证器漏评的结果不能当成通过，一律按 incomplete 处理让它进重试
    judged = {v["task_id"] for v in verdicts}
    for r in results:
        if r["task_id"] not in judged:
            verdicts.append(Verdict(
                task_id=r["task_id"], status="incomplete", completeness=0.0,
                missing_aspects=["验证器未对该结果给出结论"], contradictions=[],
                recommendation="retry", reason="验证输出里缺少该子问题的判定，按未完成处理",
            ))
    return verdicts, tokens["input"] + tokens["output"]


def coverage(tasks: list[SubTask], verdicts: list[Verdict]) -> tuple[float, float]:
    """返回 (加权完成度, 完整回答比例)。缺判定的子问题计 0。"""
    if not tasks:
        return 0.0, 0.0
    by_id = {v["task_id"]: v for v in verdicts}
    scores = [by_id[t["id"]]["completeness"] if t["id"] in by_id else 0.0 for t in tasks]
    complete = sum(1 for t in tasks if by_id.get(t["id"], {}).get("status") == "complete")
    return sum(scores) / len(tasks), complete / len(tasks)


def decide_stop(state: OrchState, completeness: float, complete_ratio: float,
                iteration: int, budget: Budget) -> tuple[bool, str]:
    """VMAO 表 2 的五条停止条件。返回 (是否停止, 原因)。原因一定会被记录。"""
    if completeness >= budget.ready_threshold:
        return True, f"完成度 {completeness:.2f} 达到阈值 {budget.ready_threshold}"
    if complete_ratio >= budget.high_confidence_coverage and completeness >= budget.high_confidence:
        return True, (f"高置信条件命中：完成度 {completeness:.2f} ≥ {budget.high_confidence}"
                      f" 且完整率 {complete_ratio:.2f} ≥ {budget.high_confidence_coverage}")
    if iteration >= budget.max_iterations:
        return True, f"达到迭代上限 {budget.max_iterations}"
    if state.get("tokens_used", 0) >= budget.token_budget:
        return True, f"达到 token 预算 {budget.token_budget}（已用 {state['tokens_used']}）"

    prev = state.get("completeness", 0.0)
    # 收益递减必须有"上一轮"可比。iteration=1 是第一轮验证，此时 prev 还是初值 0，
    # 拿它比较会在第一轮就判"没有提升"停掉，重试永远不会发生。
    # 所以要求 iteration >= 2：第一轮建基线，第二轮才是第一次真正的比较。
    if iteration >= 2 and (completeness - prev) < budget.diminishing_returns:
        return True, f"收益递减：本轮完成度提升 {completeness - prev:.3f} < {budget.diminishing_returns}"
    return False, ""


def replan(verdicts: list[Verdict], completeness: float, iteration: int,
           budget: Budget) -> tuple[list[str], int, list[str]]:
    """返回 (要重试的 id, token, 因模型漏项而强制补齐的 id)。走验证模型。

    不直接改 state —— 图里多处写 plan_warnings，改动一律通过返回值走。
    """
    retryable = [v["task_id"] for v in verdicts
                 if v["status"] != "complete" or v["recommendation"] == "retry"]
    if not retryable:
        return [], 0, []

    payload, tokens = call_json(
        REPLANNER_SYSTEM.format(ready_threshold=budget.ready_threshold,
                                iteration=iteration, max_iterations=budget.max_iterations),
        f"# 验证结论\n{json.dumps(verdicts, ensure_ascii=False)}\n\n当前整体完成度：{completeness:.2f}",
        slot=SLOT_VERIFY,
    )
    consumed = tokens["input"] + tokens["output"]
    if not isinstance(payload, dict):
        raise OrchestratorError("重规划器返回的不是 JSON 对象")

    allowed = set(retryable)
    chosen = {str(x) for x in (payload.get("retry_sub_questions") or [])} & allowed
    forced = sorted(allowed - chosen)

    # VMAO 硬性规则：不完整项必须全部重试。模型漏项时直接补齐，
    # 并把补齐动作记进 plan_warnings，不假装是模型自己决定的。
    chosen |= allowed
    return sorted(chosen), consumed, forced
