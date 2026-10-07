"""综合：把验证过的结论合成最终回答。走执行模型。

分层合成阈值来自 VMAO 6.2 节（>15K 字符或 10+ 结果时先分组摘要再整合）。
喂给综合器的只有原始任务 + 已验证的结论，不喂任何 worker 的中间对话 ——
这是 MAST FC2.3（任务脱轨 7.4%）的对策：综合器看不到跑偏的过程，就无法继承跑偏。
"""
from __future__ import annotations

import json

from .config import Budget
from .llm import OrchestratorError, call_json, call_text
from .state import SubResult, SubTask, Verdict

GROUP_SYSTEM = """你在做分层综合的第一阶段：把同一个领域的若干条已验证结论压缩成一段摘要。

要求：
- 只做压缩与去重，不要引入材料里没有的事实。
- 保留每条结论的出处（文件路径、行号、工具名）。
- 材料里标为"未完成"或"证据不足"的部分，必须原样保留这个状态，不要抹平。

只输出 JSON 对象：
{{"summary": "该领域的摘要，含关键数字与出处", "gaps": ["该领域未解决的部分"]}}
"""

FINAL_SYSTEM = """你在做多领域协作任务的最终综合。给定原始任务和若干条已验证的结论，产出一份可直接交付的回答。

结构要求：
1. 结论摘要：2-3 句，含关键数字。
2. 关键发现：5-8 条，每条必须带出处，格式 [来源 - 具体位置]。
3. 分析：2-3 段，把不同领域的发现连起来。
4. 置信度与局限：说明哪些部分证据充分、哪些证据不足或未完成。

硬性要求：
- 不允许出现材料里没有的事实、数字或结论。
- 材料里标为未完成、证据不足、或验证发现矛盾的部分，必须在第 4 部分如实写出，
  不允许为了让回答看起来完整而略过。
- 不同领域的结论如果冲突，要把冲突摆出来，不要单方面选一个。
- 如果材料里的结论缺少出处，或者出处的形式明显是编造的（比如提到不存在的工具），
  要点名指出。
"""


def _render(tasks: list[SubTask], results: list[SubResult], verdicts: list[Verdict]) -> str:
    task_map = {t["id"]: t for t in tasks}
    verdict_map = {v["task_id"]: v for v in verdicts}
    blocks = []
    for r in sorted(results, key=lambda x: x["task_id"]):
        task = task_map.get(r["task_id"], {})
        v = verdict_map.get(r["task_id"], {})
        blocks.append(json.dumps({
            "domain": r["domain"],
            "question": task.get("question", ""),
            "verification_status": v.get("status", "unverified"),
            "completeness": v.get("completeness"),
            "answer": r["text"],
            "sources": r["sources"],
            "missing_aspects": v.get("missing_aspects", []),
            "contradictions": v.get("contradictions", []),
            "error": r["error"],
        }, ensure_ascii=False))
    return "\n".join(blocks)


def needs_hierarchical(results: list[SubResult], budget: Budget) -> bool:
    return (sum(len(r["text"]) for r in results) > budget.hierarchical_synth_chars
            or len(results) > budget.hierarchical_synth_results)


def synthesize(query: str, tasks: list[SubTask], results: list[SubResult],
               verdicts: list[Verdict], *, budget: Budget) -> tuple[str, int, list[str]]:
    """返回 (最终回答, token, 使用的阶段)。"""
    if not results:
        raise OrchestratorError("没有任何执行结果可以综合")

    stages: list[str] = []
    consumed = 0

    if needs_hierarchical(results, budget):
        stages.append("hierarchical")
        by_domain: dict[str, list[SubResult]] = {}
        for r in results:
            by_domain.setdefault(r["domain"], []).append(r)

        summaries = []
        for domain, group in sorted(by_domain.items()):
            group_ids = {r["task_id"] for r in group}
            sub_tasks = [t for t in tasks if t["id"] in group_ids]
            payload, tokens = call_json(
                GROUP_SYSTEM,
                f"# 原始任务\n{query}\n\n# 领域 {domain} 的结论\n{_render(sub_tasks, group, verdicts)}")
            consumed += tokens["input"] + tokens["output"]
            if not isinstance(payload, dict):
                raise OrchestratorError(f"分层综合时领域 {domain} 的子摘要不是 JSON 对象")
            summaries.append(json.dumps(
                {"domain": domain, "summary": payload.get("summary", ""),
                 "gaps": payload.get("gaps", [])}, ensure_ascii=False))
        material = "\n".join(summaries)
    else:
        stages.append("direct")
        material = _render(tasks, results, verdicts)

    answer, tokens = call_text(FINAL_SYSTEM, f"# 原始任务\n{query}\n\n# 已验证的结论\n{material}")
    consumed += tokens["input"] + tokens["output"]
    return answer.strip(), consumed, stages
