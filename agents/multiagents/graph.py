"""LangGraph 编排图。

拓扑（VMAO 的 Plan → Execute → Verify → Replan → Synthesize）：

    START → plan → [Send 扇出 → worker*] → join → verify → (重试 | synthesize) → END

两条必须遵守的 LangGraph 结构约束（都是上一轮实测踩出来的）：

1. **条件边不能直接挂在 Send 的目标节点上。**
   `worker` 是被 Send 扇出的。如果条件边挂在它上面，LangGraph 会为每个完成的
   worker 实例各求值一次路由函数，而且拿到的是本超步写入之前的旧状态 ——
   同一批 Send 会被并集多次。上一轮实测 4 个子任务被派发 15 次。
   所以中间插一个普通节点 `join`，worker 只接普通边到 join，条件边挂在 join 上。

2. **并行节点写入的通道必须有 reducer。** 见 state.py 的说明。

`dispatch`（选批）与 `replan`（定重试）是**纯函数**，由条件边调用，不做成节点：
它们是决策不是执行，混进节点会让边的语义不清。
"""
from __future__ import annotations

import time
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from . import trace
from .config import Budget
from .domains import get_domain
from .llm import model_config
from .planning import plan
from .scheduling import (build_dep_context, check_before_synthesis, ready_tasks,
                         select_batch)
from .state import OrchState, WorkerPayload, latest_results, unverified_results
from .synthesis import synthesize
from .verification import coverage, decide_stop, replan, verify
from .worker import run_worker


def make_graph(budget: Budget):
    """构建并编译编排图。"""
    cfg = model_config()

    def _tok(model_name: str, tokens: int) -> dict[str, int]:
        return {model_name: tokens} if tokens else {}

    # ── plan：拆解 + 清洗。守 MAST FC1.1（违反任务规范 11.8%）────────
    async def plan_node(state: OrchState) -> dict[str, Any]:
        started = time.monotonic()
        tasks, warnings, tokens = plan(state["query"], budget=budget)
        trace.node("plan", f"拆出 {len(tasks)} 个子问题  tok={tokens}  "
                           f"{time.monotonic() - started:.1f}s  {cfg.exec_model}")
        for t in tasks:
            trace.detail(f"{t['id']} [{t['domain']}] pri={t['priority']} "
                         f"deps={trace.ids(t['dependencies'])}  "
                         f"{trace.truncate(t['question'], 80)}")
        for w in warnings:
            trace.detail(f"警告：{w}")
        return {
            "tasks": tasks,
            "plan_warnings": warnings,
            "tokens_used": tokens,
            "tokens_by_model": _tok(cfg.exec_model, tokens),
            "events": [{
                "node": "plan", "model": cfg.exec_model,
                "seconds": round(time.monotonic() - started, 2),
                "tasks": [{"id": t["id"], "domain": t["domain"], "deps": t["dependencies"],
                           "priority": t["priority"]} for t in tasks],
                "warnings": warnings, "tokens": tokens,
            }],
        }

    # ── worker：Send 扇出的目标，只接普通边到 join ──────────────────
    async def worker_node(payload: WorkerPayload) -> dict[str, Any]:
        task = payload["task"]
        domain = get_domain(task["domain"])
        attempt = payload["attempt"]
        started = time.monotonic()

        result, intercepts = await run_worker(
            task, domain, payload["query"], payload["dep_context"],
            attempt=attempt, budget=budget)

        elapsed = time.monotonic() - started
        used = result["input_tokens"] + result["output_tokens"]
        for it in intercepts:
            it["task_id"] = task["id"]

        trace.node("worker", f"{task['id']} [{domain.name}] attempt={attempt} "
                             f"{'ok' if result['ok'] else 'FAIL'} {elapsed:.1f}s "
                             f"tok={used}"
                             + (f" 净化-{result['purged_chars']}字"
                                if result.get("purged_chars") else ""))
        if result["error"]:
            trace.detail(f"{task['id']} error: {trace.truncate(result['error'], 150)}")
        if payload["dep_context"]:
            trace.detail(f"{task['id']} 注入了上游依赖结果 "
                         f"{len(payload['dep_context'])} 字符")

        return {
            "results": [result],
            "intercepts": intercepts,
            "tokens_used": used,
            "tokens_by_model": _tok(cfg.exec_model, used),
            "attempts": {task["id"]: attempt},
            "events": [{
                "node": "worker", "task_id": task["id"], "domain": domain.name,
                "engine": result["engine"], "attempt": attempt, "ok": result["ok"],
                "model": cfg.exec_model,
                "seconds": round(elapsed, 2), "tokens": used,
                "purified": result.get("purified", False),
                "purged_chars": result.get("purged_chars", 0),
                "error": result["error"],
                "start": started, "end": time.monotonic(),
            }],
        }

    # ── join：汇合点。功能接近空操作，但它是「路由函数每超步只求值一次」的保证 ──
    async def join_node(state: OrchState) -> dict[str, Any]:
        done = sorted(latest_results(state))
        pending = sorted(t["id"] for t in state["tasks"] if t["id"] not in set(done))
        unverified = sorted(r["task_id"] for r in unverified_results(state))
        trace.node("join", f"完成 {len(done)}/{len(state['tasks'])}  "
                           f"待办 {trace.ids(pending)}  未验证 {trace.ids(unverified)}")
        return {"events": [{
            "node": "join", "done": done, "pending": pending,
            "unverified": unverified,
        }]}

    # ── verify：独立验证（走验证模型）+ 停止判断 + 重规划。
    #    守 MAST FC3.2（无验证 8.2%）与 FC3.3（错误验证 9.1%）────────
    async def verify_node(state: OrchState) -> dict[str, Any]:
        started = time.monotonic()
        tasks, results = state["tasks"], state["results"]
        iteration = state["iteration"] + 1

        verdicts, tokens = verify(state["query"], tasks, results)
        merged = _merge(state.get("verdicts") or [], verdicts)
        completeness, complete_ratio = coverage(tasks, merged)

        stop, reason = decide_stop(state, completeness, complete_ratio, iteration, budget)
        retry_ids: list[str] = []
        forced: list[str] = []
        if not stop:
            retry_ids, replan_tokens, forced = replan(verdicts, completeness, iteration, budget)
            tokens += replan_tokens
            if not retry_ids:
                stop, reason = True, ("验证标记了不完整项但无可重试子任务，"
                                      "直接进入综合（缺口会在结论里如实列出）")

        warnings = list(state.get("plan_warnings") or [])
        if forced:
            warnings.append(f"重规划器漏报 {forced}，已按 VMAO 规则强制加入重试")

        n_complete = sum(1 for v in verdicts if v["status"] == "complete")
        n_contra = sum(len(v["contradictions"]) for v in verdicts)
        trace.node("verify", f"第 {iteration} 轮  完成度 {completeness:.2f}  "
                             f"({n_complete}/{len(verdicts)} complete)  矛盾 {n_contra}  "
                             f"tok={tokens}  {time.monotonic() - started:.1f}s  "
                             f"{cfg.verify_model}")
        for v in verdicts:
            trace.detail(f"{v['task_id']} {v['status']:<10} {v['completeness']:.2f} "
                         f"缺失{len(v['missing_aspects'])} 矛盾{len(v['contradictions'])} "
                         f"建议={v['recommendation']}")
            for c in v["contradictions"][:2]:
                trace.detail(f"  {v['task_id']} 矛盾：{trace.truncate(c, 130)}")
        if retry_ids:
            trace.detail(f"判定需重试：{trace.ids(retry_ids)}"
                         + (f"（其中 {trace.ids(forced)} 是模型漏报后强制的）" if forced else ""))
        trace.detail(f"停止={stop}  {trace.truncate(reason, 120)}" if stop
                     else f"未停止，继续第 {iteration + 1} 轮派发")

        return {
            "verdicts": verdicts,
            "iteration": iteration,
            "completeness": completeness,
            "retry_ids": retry_ids,
            "tokens_used": tokens,
            "tokens_by_model": _tok(cfg.verify_model, tokens),
            "plan_warnings": warnings,
            "verified": {r["task_id"]: r["attempt"] for r in results},
            "stop_reason": reason if stop else "",
            "events": [{
                "node": "verify", "iteration": iteration, "model": cfg.verify_model,
                "completeness": round(completeness, 3),
                "complete_ratio": round(complete_ratio, 3),
                "retry_ids": retry_ids, "forced_retries": forced,
                "stop": stop, "stop_reason": reason, "tokens": tokens,
                "seconds": round(time.monotonic() - started, 2),
                "verdicts": [{"task_id": v["task_id"], "status": v["status"],
                              "completeness": v["completeness"],
                              "recommendation": v["recommendation"],
                              "contradictions": v["contradictions"]} for v in verdicts],
            }],
        }

    # ── synthesize：只吃原始任务 + 已验证结论。守 MAST FC2.3（任务脱轨 7.4%）──
    async def synth_node(state: OrchState) -> dict[str, Any]:
        started = time.monotonic()
        violations = check_before_synthesis(state)
        answer, tokens, stages = synthesize(
            state["query"], state["tasks"], state["results"], state["verdicts"], budget=budget)
        trace.node("synthesize", f"阶段={trace.ids(stages)}  输入 "
                                 f"{len(state['results'])} 条结果  tok={tokens}  "
                                 f"{time.monotonic() - started:.1f}s  {cfg.exec_model}")
        trace.detail(f"结论 {len(answer)} 字符")
        if violations:
            trace.detail(f"进综合前的不变量检查：{violations}")
        return {
            "final_answer": answer,
            "tokens_used": tokens,
            "tokens_by_model": _tok(cfg.exec_model, tokens),
            "events": [{
                "node": "synthesize", "model": cfg.exec_model, "stages": stages,
                "tokens": tokens, "seconds": round(time.monotonic() - started, 2),
                "invariant_violations": violations,
            }],
        }

    # ── 路由：唯一的派发决策点。纯函数，不调模型，不产生副作用 ──────────
    def route(state: OrchState) -> list[Send] | str:
        if state.get("stop_reason"):
            trace.node("route", f"已停止（{trace.truncate(state['stop_reason'], 80)}）→ synthesize")
            return "synthesize"

        results = latest_results(state)
        ready = ready_tasks(
            state["tasks"], results, state.get("attempts") or {},
            state.get("retry_ids") or [], state.get("iteration", 0), budget)
        batch = select_batch(ready, budget)
        if batch:
            # 不写"第 N 轮"：state["iteration"] 是**已完成的验证轮数**，首轮派发时是 0，
            # 打给用户看会以为"从第 0 轮开始"。轮次信息在 verify 那行已经有了。
            trace.node("route", f"派发 {len(batch)} 个 → {trace.ids(t['id'] for t in batch)}"
                                f"（就绪 {len(ready)} 个，并发上限 {budget.max_concurrent}）")
            sends: list[Send] = []
            for task in batch:
                sends.append(Send("worker", WorkerPayload(
                    task=task,
                    query=state["query"],
                    dep_context=build_dep_context(task, results),
                    attempt=(state.get("attempts") or {}).get(task["id"], 0) + 1,
                )))
            return sends

        pending = unverified_results(state)
        if pending:
            trace.node("route", f"无待派发，{len(pending)} 条结果未验证 → verify")
            return "verify"
        trace.node("route", "无待派发且无未验证 → synthesize")
        return "synthesize"

    builder = StateGraph(OrchState)
    builder.add_node("plan", plan_node)
    builder.add_node("worker", worker_node)
    builder.add_node("join", join_node)
    builder.add_node("verify", verify_node)
    builder.add_node("synthesize", synth_node)

    builder.add_edge(START, "plan")
    # plan / join / verify 每超步只执行一次，条件边挂在这三个上是安全的
    for source in ("plan", "join", "verify"):
        builder.add_conditional_edges(source, route, ["worker", "verify", "synthesize"])
    # worker 是 Send 目标，只接普通边到 join —— 绝不直接挂条件边
    builder.add_edge("worker", "join")
    builder.add_edge("synthesize", END)

    return builder.compile()


def _merge(existing: list[dict], new: list[dict]) -> list[dict]:
    by_id = {v["task_id"]: v for v in list(existing) + list(new)}
    return sorted(by_id.values(), key=lambda v: v["task_id"])
