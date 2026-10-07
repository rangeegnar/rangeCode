"""调度与不变量 —— 全部是纯函数。

这个模块是整张图最容易出错的地方，也是上一轮唯一真正出过严重 bug 的地方
（4 个子任务被派发 15 次）。所以它不调模型、不依赖运行时，可以被穷举测试。

核心不变量（每次运行结束都检查）：
  I1  每个 (task_id, attempt) 组合最多被派发一次
  I2  任何子任务的派发次数 <= max_iterations
  I2b attempts 的记录值必须等于实际执行次数
  I3  进入综合阶段时 retry_ids 必须为空
  I4  所有结果都被验证到自己的 attempt
  I5  tokens_used 不超过预算
"""
from __future__ import annotations

from typing import Any

from .config import Budget
from .state import OrchState, SubResult, SubTask


def max_attempts(budget: Budget) -> int:
    """允许的最大派发次数。

    语义对齐 verify 的停止条件：第 `max_iterations` 轮验证后必然停止
    （`iteration >= max_iterations`），而只有第 1..max_iterations-1 轮会触发重试。
    所以最多派发 max_iterations 次。

    下限取 1：max_iterations=0 时也要允许首轮执行，否则第一个子任务都派不出去。
    """
    return max(1, budget.max_iterations)


def ready_tasks(
    tasks: list[SubTask],
    results: dict[str, SubResult],
    attempts: dict[str, int],
    retry_ids: list[str],
    iteration: int,
    budget: Budget,
) -> list[SubTask]:
    """挑出当前可以派发的子任务，按优先级降序、id 升序排列。

    三个条件同时满足才可派发：
      1. 依赖都已完成
      2. 派发次数未到上限，且本轮（iteration）还没派发过
      3. 首轮（attempts==0），或者被本轮验证判为需要重试

    条件 2 里的 `used > iteration` 是防重复派发的关键：重试轮 N 只允许把某个
    子任务从 attempt N 推到 N+1，不允许在同一轮里反复推。
    """
    done = set(results)
    retry = set(retry_ids or [])
    limit = max_attempts(budget)

    out: list[SubTask] = []
    for task in tasks:
        used = attempts.get(task["id"], 0)
        if used >= limit:
            continue
        if used > iteration:
            continue
        if not all(dep in done for dep in task["dependencies"]):
            continue
        if used == 0:
            out.append(task)
        elif task["id"] in retry:
            out.append(task)

    out.sort(key=lambda t: (-t["priority"], t["id"]))
    return out


def select_batch(ready: list[SubTask], budget: Budget) -> list[SubTask]:
    return ready[: budget.max_concurrent]


def build_dep_context(task: SubTask, results: dict[str, SubResult]) -> str:
    """拼装上游依赖结果。

    VMAO 把这一步做成可选项（context_from_deps）。这里改成只要有依赖就一定注入：
    子问题数量有上限、依赖结果是 worker 的结论（而非原始工具输出）体量可控，
    省掉这个开关就少一条"有依赖却没拿到上游结果"的失败路径。
    """
    if not task["dependencies"]:
        return ""
    parts: list[str] = []
    for dep in task["dependencies"]:
        res = results.get(dep)
        if res and res.get("text"):
            parts.append(f"## 来自 {dep}（{res['domain']}）\n{res['text']}")
    return "\n\n".join(parts)


def check_invariants(state: OrchState, budget: Budget) -> list[str]:
    """返回被破坏的不变量列表。空列表表示全部通过。

    不吞异常、不给默认通过 —— 调用方必须把非空结果当成失败处理。
    """
    violations: list[str] = []

    worker_events = [e for e in state.get("events", []) if e.get("node") == "worker"]
    counts: dict[tuple[str, int], int] = {}
    for e in worker_events:
        key = (str(e.get("task_id")), int(e.get("attempt", 0)))
        counts[key] = counts.get(key, 0) + 1

    # I1：同一个 (task, attempt) 只能出现一次
    dupes = {k: v for k, v in counts.items() if v > 1}
    if dupes:
        violations.append(f"I1 违反：同一 (task_id, attempt) 被重复派发 {dupes}")

    # I2：派发次数上限
    limit = max_attempts(budget)
    attempts = state.get("attempts") or {}
    over = {k: v for k, v in attempts.items() if v > limit}
    if over:
        violations.append(f"I2 违反：派发次数超过上限 {limit}：{over}")

    # I2b：记录到的派发次数必须与实际执行次数一致
    actual: dict[str, int] = {}
    for (tid, _), n in counts.items():
        actual[tid] = actual.get(tid, 0) + n
    mismatch = {
        tid: {"记录": attempts.get(tid, 0), "实际": n}
        for tid, n in actual.items()
        if attempts.get(tid, 0) != n
    }
    if mismatch:
        violations.append(f"I2b 违反：attempts 记录与实际执行次数不一致 {mismatch}")

    # I4：进入综合时所有结果都必须被验证到自己的 attempt
    verified = state.get("verified") or {}
    unverified = [r["task_id"] for r in state.get("results", [])
                  if verified.get(r["task_id"], 0) < r["attempt"]]
    if unverified and state.get("final_answer"):
        violations.append(f"I4 违反：进入综合时仍有未验证结果 {unverified}")

    # I5：预算
    if state.get("tokens_used", 0) > budget.token_budget:
        violations.append(f"I5 违反：token {state['tokens_used']} 超过预算 {budget.token_budget}")

    return violations


def check_before_synthesis(state: OrchState) -> list[str]:
    """I3 单独拆出来：只在真正要进综合那一刻检查。"""
    if state.get("retry_ids"):
        return [f"I3 违反：retry_ids 非空 {state['retry_ids']} 时进入综合"]
    return []


def summarise(state: OrchState) -> dict[str, Any]:
    """给运行记录用的一页摘要。"""
    verified = state.get("verified") or {}
    return {
        "tokens_used": state.get("tokens_used", 0),
        "tokens_by_model": dict(state.get("tokens_by_model") or {}),
        "iteration": state.get("iteration", 0),
        "completeness": state.get("completeness", 0.0),
        "stop_reason": state.get("stop_reason", ""),
        "tasks": len(state.get("tasks", [])),
        "dispatches": len([e for e in state.get("events", []) if e.get("node") == "worker"]),
        "attempts": dict(state.get("attempts") or {}),
        "unverified": [r["task_id"] for r in state.get("results", [])
                       if verified.get(r["task_id"], 0) < r["attempt"]],
        "intercepts": len(state.get("intercepts") or []),
    }
