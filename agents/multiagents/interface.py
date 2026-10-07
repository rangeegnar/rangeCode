"""编排层入口 —— 给 agents/agent.py 的 agent-loop 调用的那一个函数。

对 `agents/` 的接入是三处新增，不动任何现有逻辑：

1) `agents/agent.py::__init__`（工具集赋值之后）注册工具定义：
       from agents.multiagents.definition import MULTIAGENT_TOOL_DEF
       self.tools = self.tools + [MULTIAGENT_TOOL_DEF]

   只 import `definition`，**不要 import 本模块** —— 否则每构造一个 Agent
   都会把 langgraph 拉进来。

2) `agents/agent.py::_execute_tool_call` 加一条分支（延迟 import）：
       if name == "multiagent":
           from agents.multiagents.interface import execute_multiagent
           return await execute_multiagent(
               inp, session_id=self.session_id,
               host_model=self.model, host_api_base=self.api_base,
               host_api_key=self.api_key, host_embedding_model=self.embedding_model)

3) `agents/tools.py::check_permission` 加一条 plan 模式守卫：
       if mode == "plan" and tool_name == "multiagent":
           return {"action": "deny", "message": "Blocked in plan mode: multiagent"}

   为什么第 3 条不能省：`Agent.toggle_plan_mode()` 运行时不重筛 `self.tools`，
   进入 plan 模式后工具仍在表里；而 `check_permission` 对未知工具名会落到
   最后一行 `{"action": "allow"}`，plan 模式只拦 EDIT_TOOLS 和 run_shell。

不要标进 `CONCURRENCY_SAFE_TOOLS`：编排层内部已经扇出到端点并发上限
（max_concurrent=2，取 3 实测撞 429），同一轮再叠加并发会直接 429。
"""
from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

from . import llm, records, trace
from .config import DEFAULT_BUDGET, Budget
from .definition import MULTIAGENT_TOOL_DEF
from .domains import ENGINE_LG_TOOL
from .llm import apply_host_model
from .runner import run_orchestration

__all__ = ["MULTIAGENT_TOOL_DEF", "execute_multiagent", "execute_multiagent_with_usage",
           "format_tool_result"]


def _meta_block(record: dict[str, Any]) -> list[str]:
    """可信度元信息。**必须排在结论之前**，理由见 format_tool_result。"""
    s = record.get("summary") or {}
    models = record.get("models") or {}
    lines = [
        f"[编排结果元信息] 完成度 {s.get('completeness', 0.0):.2f}"
        f" | 迭代 {s.get('iteration', 0)} 轮"
        f" | 派发 {s.get('dispatches', 0)} 次"
        f" | token {s.get('tokens_used', 0)}"
        f" | 耗时 {record.get('wall_seconds', 0)}s",
        f"停止原因：{s.get('stop_reason') or '（未记录）'}",
        f"模型：执行 {models.get('exec')} / 验证 {models.get('verify')}"
        + ("" if models.get("verify_is_independent") else "（两者相同，验证结论可能共享偏差）"),
    ]

    by_model = record.get("cost_by_model") or {}
    if by_model:
        lines.append("token 按模型拆分：" + "，".join(f"{m}={t}" for m, t in sorted(by_model.items())))

    unfinished = [v for v in record.get("verdicts", []) if v.get("status") != "complete"]
    if unfinished:
        lines.append("未完成或部分完成的子问题：")
        for v in unfinished:
            miss = "；".join(v.get("missing_aspects") or []) or "未说明"
            lines.append(f"  - {v['task_id']}（{v['status']}，完成度 {v['completeness']:.2f}）：{miss}")

    contradictions = [(v["task_id"], c) for v in record.get("verdicts", [])
                      for c in (v.get("contradictions") or [])]
    if contradictions:
        lines.append("验证发现的矛盾（结论中若有相反口径，以此为准）：")
        for tid, c in contradictions:
            lines.append(f"  - {tid}：{c}")

    failures = [r for r in record.get("results", []) if r.get("error")]
    if failures:
        lines.append("执行失败项：")
        for r in failures:
            lines.append(f"  - {r['task_id']}（{r['engine']}）：{r['error']}")

    for w in record.get("plan_warnings") or []:
        lines.append(f"规划警告：{w}")

    for v in record.get("invariant_violations") or []:
        lines.append(f"不变量被破坏（这是实现缺陷，不是内容缺口）：{v}")

    purged = sum(int(i.get("purged_chars") or 0) for i in record.get("intercepts", []))
    if purged:
        lines.append(f"监督层净化：{len(record.get('intercepts', []))} 次干预，累计省下 {purged} 字符")

    mismatch = record.get("accounting_mismatch")
    if mismatch:
        lines.append(f"token 记账不一致（实现缺陷，不是内容缺口）：按事件合计差 {mismatch}")

    if record.get("record_path"):
        lines.append(f"完整轨迹：{record['record_path']}")
    return lines


def format_tool_result(record: dict[str, Any]) -> str:
    """工具返回文本。

    **元信息在前，结论在后。** 这不是排版偏好，是必须的：主 agent-loop 会调
    `Agent._persist_large_result`（阈值 30KB），超过就只保留前 200 行、其余落盘。
    如果把结论放前面，一旦超阈值，"完成度 / 未完成项 / 矛盾 / 失败项"这些
    说明结论有多不可信的信息会被整段截掉，主 agent 只看到一段看起来完整的
    文字然后照信 —— 整个双模型验证要防的就是这件事，会被最后一个环节抹平。
    """
    lines = _meta_block(record)
    lines += ["", "---", "", "【结论】", record.get("final_answer", "").strip() or "(编排未产出结论)"]
    return "\n".join(lines)


def _usage_from(record: dict[str, Any]) -> dict[str, Any]:
    """把这次运行的 token 拆成 input / output，供宿主 Agent 精确记账。

    为什么能精确：
      - 编排层节点（plan / verify / replan / synthesize）与 `plain-llm` 的 worker
        全都走 `llm.call_json` / `llm.call_text`，`llm` 模块里按 input/output 分别累加；
      - 唯一漏网的是 `lg-tool-agent` 的 worker —— 它由 create_agent 内部自己建
        ChatOpenAI，不走上面那条路，但它的 SubResult 上带着 input_tokens / output_tokens。

    所以这里是两份精确数据相加，不是估算。

    `accounting_mismatch` 是自己给自己上的检查：它必须等于 0（两份之和 == 事件合计）。
    不为 0 说明有一处调用没被记到，会写进返回文本而不是被吞掉。

    `Agent._get_current_cost_usd` 按 input $3/M、output $15/M 计价，所以拆分必须准确，
    不能把总量一股脑塞进 input。
    """
    counters = llm.usage_snapshot()
    in_tok = int(counters.get("input", 0))
    out_tok = int(counters.get("output", 0))

    for r in record.get("results", []):
        if r.get("engine") == ENGINE_LG_TOOL:
            in_tok += int(r.get("input_tokens") or 0)
            out_tok += int(r.get("output_tokens") or 0)

    total = in_tok + out_tok
    reported = int((record.get("summary") or {}).get("tokens_used") or 0)
    return {
        "input": in_tok,
        "output": out_tok,
        "total": total,
        "by_model": dict(record.get("cost_by_model") or {}),
        "reported_total": reported,
        "accounting_mismatch": total - reported,
    }


async def _execute(
    inp: dict[str, Any],
    *,
    session_id: str | None,
    budget: Budget,
    host_model: str | None,
    host_api_base: str | None,
    host_api_key: str | None,
    host_embedding_model: str | None,
    progress: bool = False,
    records_root_override: Any,
) -> tuple[str, dict[str, Any]]:
    """跑一次编排。返回 (给主 agent 的文本, 用量拆分)。

    参数非法时返回 `("Error: ...", {})`，且**不建目录、不写记录**。

    `progress=True` 会把节点进度按宿主 `agents/ui.py` 的样式打出来
    （plan / route / worker / join / verify / synthesize 各一行）。默认关：
    脚本直连和单元测试不该被刷屏。
    """
    task = str(inp.get("task") or "").strip()
    if not task:
        return "Error: multiagent requires a non-empty 'task'.", {}

    if inp.get("max_iterations") is not None:
        try:
            iterations = max(0, int(inp["max_iterations"]))
        except (TypeError, ValueError):
            return f"Error: max_iterations must be an integer, got {inp['max_iterations']!r}", {}
        budget = replace(budget, max_iterations=iterations)

    if progress:
        trace.set_enabled(True)
        trace.use_host_ui()
        trace.reset()
    else:
        trace.set_enabled(False)

    apply_host_model(
        host_model=host_model, host_api_base=host_api_base,
        host_api_key=host_api_key, host_embedding_model=host_embedding_model,
    )
    llm.reset_usage()

    # 先建目录再跑图：中途崩溃/被中断时，目录和已发生的事实都还在
    directory = records.session_dir(session_id, root=records_root_override)

    started = time.monotonic()
    try:
        record = await run_orchestration(task, budget=budget)
    except Exception as exc:
        records.save_failure(
            session_id=session_id, task=task, error=exc,
            wall_seconds=time.monotonic() - started, budget=budget,
            root=records_root_override)
        raise

    record["session_id"] = session_id
    record["budget"] = {
        "max_concurrent": budget.max_concurrent,
        "max_iterations": budget.max_iterations,
        "ready_threshold": budget.ready_threshold,
        "token_budget": budget.token_budget,
        "tau_len": budget.tau_len,
        "enable_purification": budget.enable_purification,
    }
    path = records.save_run(record, session_id=session_id, root=records_root_override)
    record["record_path"] = str(path)

    # 记录必须落在本次会话的目录里。写错目录属于实现缺陷，明确报错，不静默接受。
    if path.parent != directory:
        raise RuntimeError(f"运行记录写到了 {path.parent}，与本次会话目录 {directory} 不一致")

    usage = _usage_from(record)
    if usage["accounting_mismatch"]:
        # 不让它静默：写进返回文本，主 agent 和人都能看到记账对不上
        record["accounting_mismatch"] = usage["accounting_mismatch"]
    return format_tool_result(record), usage


async def execute_multiagent(
    inp: dict[str, Any],
    *,
    session_id: str | None = None,
    budget: Budget = DEFAULT_BUDGET,
    host_model: str | None = None,
    host_api_base: str | None = None,
    host_api_key: str | None = None,
    host_embedding_model: str | None = None,
    progress: bool = False,
    records_root_override: Any = None,
) -> str:
    """执行一次多智能体编排，返回给主 agent 的文本（工具契约就是返回字符串）。

    `session_id` 由调用方（`_execute_tool_call`）传入 `self.session_id`，用来隔离
    运行记录目录。为空则落到 `_manual` 子目录，不静默丢弃。

    宿主模型参数（`host_*`）让执行槽位跟随主 agent 的模型，而不是自己去读 .env ——
    用户在 CLI 用 `--model` 覆盖时，编排层必须跟得上。验证槽位始终来自 .env 的
    MODEL4TEST，两者不等才叫"独立验证"。

    `progress=True` 打印节点进度（走宿主 UI 样式）。
    `records_root_override` 只给测试用。
    """
    text, _usage = await _execute(
        inp, session_id=session_id, budget=budget, host_model=host_model,
        host_api_base=host_api_base, host_api_key=host_api_key,
        host_embedding_model=host_embedding_model, progress=progress,
        records_root_override=records_root_override)
    return text


async def execute_multiagent_with_usage(
    inp: dict[str, Any],
    *,
    session_id: str | None = None,
    budget: Budget = DEFAULT_BUDGET,
    host_model: str | None = None,
    host_api_base: str | None = None,
    host_api_key: str | None = None,
    host_embedding_model: str | None = None,
    progress: bool = False,
    records_root_override: Any = None,
) -> tuple[str, dict[str, Any]]:
    """同上，但额外返回用量拆分，供宿主 Agent 把 token 记进自己的计数器。

    `agents/agent.py::_execute_multiagent_tool` 用这个版本 —— 否则那十几万 token
    不会出现在 `show_cost()` 里，也进不了预算门禁。参数非法或未跑起来时
    usage 为空 dict。
    """
    return await _execute(
        inp, session_id=session_id, budget=budget, host_model=host_model,
        host_api_base=host_api_base, host_api_key=host_api_key,
        host_embedding_model=host_embedding_model, progress=progress,
        records_root_override=records_root_override)
