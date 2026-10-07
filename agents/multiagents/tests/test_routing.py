"""调度的穷举测试 —— 全程零模型调用。

为什么先做这个：上一轮唯一真正严重的 bug（4 个子任务被派发 15 次）是纯逻辑错误，
根本不需要调模型就能发现。当时却花了四次真模型运行、几十万 token 才定位到它。

这里用假后端 + 假引擎把整张图跑起来，只断言结构性质：
    每个 (task_id, attempt) 最多派发一次；派发次数 == 预期；不变量全部通过。

跑法：
    python -m pytest tests/ -q
"""
from __future__ import annotations

import asyncio
from typing import Any

from agents.multiagents import worker as worker_mod
from agents.multiagents.config import Budget, with_overrides
from agents.multiagents.graph import make_graph
from agents.multiagents.llm import SLOT_VERIFY, set_backend
from agents.multiagents.scheduling import check_invariants, max_attempts
from agents.multiagents.state import initial_state


# ── 假后端 ────────────────────────────────────────────────
class FakeBackend:
    """按系统提示词特征分派：规划 / 验证 / 重规划。记录每个槽位被调用了几次。"""

    independent = True

    def __init__(self, plan_obj: dict[str, Any], verdicts: list[dict[str, Any]] | None = None,
                 retry_plan: list[list[str]] | None = None):
        self.plan_obj = plan_obj
        self.verdicts = verdicts or []
        self.retry_plan = retry_plan or [[]]
        self.retry_calls = 0
        self.calls: list[tuple[str, str]] = []      # (角色, 槽位)

    def call_json(self, system: str, user: str, max_tokens: int | None = None, **kwargs):
        slot = kwargs.get("slot", "exec")
        if "规划器" in system:
            self.calls.append(("plan", slot))
            return dict(self.plan_obj), {"input": 1, "output": 1}
        if "独立验证者" in system:
            self.calls.append(("verify", slot))
            return {"verdicts": list(self.verdicts)}, {"input": 1, "output": 1}
        if "重规划者" in system:
            self.calls.append(("replan", slot))
            idx = min(self.retry_calls, len(self.retry_plan) - 1)
            self.retry_calls += 1
            return ({"retry_sub_questions": list(self.retry_plan[idx]),
                     "new_sub_questions": [], "explanation": "test"},
                    {"input": 1, "output": 1})
        raise AssertionError(f"假后端收到未知 prompt：{system[:80]}")

    def call_text(self, system: str, user: str, **kwargs):
        self.calls.append(("synth", kwargs.get("slot", "exec")))
        return "（测试用最终结论）", {"input": 1, "output": 1}


def make_fake_engine(fail_task_ids: set[str] | None = None):
    fail_task_ids = fail_task_ids or set()
    seen: list[tuple[str, int]] = []

    async def _engine(task, domain, query, dep_context, attempt, supervisor, budget):
        seen.append((task["id"], attempt))
        ok = task["id"] not in fail_task_ids
        return {
            "task_id": task["id"], "domain": domain.name, "engine": domain.engine,
            "ok": ok, "text": f"结论 {task['id']}" if ok else "",
            "sources": [f"{task['id']}.py:1"] if ok else [],
            "input_tokens": 10, "output_tokens": 5,
            "error": None if ok else "测试注入的失败",
            "attempt": attempt, "purified": False, "purged_chars": 0,
        }

    _engine.seen = seen  # type: ignore[attr-defined]
    return _engine


def run_graph(plan_obj: dict, *, verdicts=None, retry_plan=None, budget=None,
              fail_task_ids=None, independent=True) -> tuple[dict, list[tuple[str, int]], FakeBackend]:
    budget = budget or Budget()
    engine = make_fake_engine(fail_task_ids)
    fake = FakeBackend(plan_obj, verdicts, retry_plan)
    fake.independent = independent
    set_backend(fake)
    worker_mod.set_engine_override(engine)
    try:
        graph = make_graph(budget)
        final = asyncio.run(graph.ainvoke(
            initial_state("测试任务"),
            config={"recursion_limit": budget.graph_recursion_limit}))
    finally:
        worker_mod.set_engine_override(None)
        set_backend(None)
    return final, engine.seen, fake  # type: ignore[attr-defined]


def task(tid: str, domain: str = "code", deps=None, priority: int = 5) -> dict:
    return {"id": tid, "question": f"问题 {tid}", "domain": domain,
            "dependencies": deps or [], "priority": priority,
            "context_from_deps": False, "verification_criteria": f"能回答 {tid}"}


def plan_obj(*tasks: dict) -> dict:
    return {"sub_questions": list(tasks), "explanation": "测试计划"}


def all_complete(tasks: list[dict]) -> list[dict]:
    return [{"task_id": t["id"], "status": "complete", "completeness": 1.0,
             "missing_aspects": [], "contradictions": [], "recommendation": "accept",
             "reason": "测试判定"} for t in tasks]


def all_incomplete(tasks: list[dict]) -> list[dict]:
    return [{"task_id": t["id"], "status": "incomplete", "completeness": 0.0,
             "missing_aspects": ["测试缺口"], "contradictions": [], "recommendation": "retry",
             "reason": "测试判定"} for t in tasks]


# ── 结构不变量 ─────────────────────────────────────────────

def test_single_task_dispatched_once():
    t = [task("sq_001")]
    final, seen, _ = run_graph(plan_obj(*t), verdicts=all_complete(t))
    assert seen == [("sq_001", 1)], seen
    assert check_invariants(final, Budget()) == []
    assert final["final_answer"]


def test_three_independent_tasks_no_duplicate_dispatch():
    """核心用例：上一轮这里 3 个任务被派发 9 次。"""
    t = [task("sq_001", priority=10), task("sq_002", priority=9), task("sq_003", priority=8)]
    final, seen, _ = run_graph(plan_obj(*t), verdicts=all_complete(t))
    assert sorted(seen) == [("sq_001", 1), ("sq_002", 1), ("sq_003", 1)], seen
    assert len(seen) == 3
    assert check_invariants(final, Budget()) == []


def test_max_concurrent_splits_into_waves():
    t = [task("sq_001", priority=10), task("sq_002", priority=9), task("sq_003", priority=8)]
    budget = with_overrides(Budget(), max_concurrent=2)
    final, seen, _ = run_graph(plan_obj(*t), verdicts=all_complete(t), budget=budget)
    assert len(seen) == 3, seen
    assert sorted(x[0] for x in seen) == ["sq_001", "sq_002", "sq_003"]
    assert check_invariants(final, budget) == []


def test_dependency_orders_execution():
    t = [task("sq_001", priority=1), task("sq_002", deps=["sq_001"], priority=10)]
    final, seen, _ = run_graph(plan_obj(*t), verdicts=all_complete(t))
    assert seen == [("sq_001", 1), ("sq_002", 1)], seen
    assert check_invariants(final, Budget()) == []


def test_retry_actually_happens():
    """重试必须真的发生。

    这条测试是写出来之后才暴露出一个真实逻辑错误：收益递减判定原本写成
    `iteration >= 1`，而第一轮验证时 prev 还是初值 0、completeness 也接近 0，
    于是第一轮就直接判"没有提升"停掉，重试永远不会发生。
    改成 `iteration >= 2`（第一轮建基线，第二轮才比较）后才对。
    """
    t = [task("sq_001")]
    budget = with_overrides(Budget(), max_iterations=3)
    final, seen, _ = run_graph(plan_obj(*t), verdicts=all_incomplete(t), budget=budget)

    assert [att for _, att in seen] == [1, 2], seen
    assert final["attempts"]["sq_001"] == 2
    assert "收益递减" in final["stop_reason"], final["stop_reason"]
    assert check_invariants(final, budget) == []


def test_no_duplicate_task_attempt_ever():
    """这条直接对着上一轮的 bug：同一个 (task, attempt) 被派发多次。"""
    t = [task("sq_001", priority=10), task("sq_002", priority=9), task("sq_003", priority=8)]
    budget = with_overrides(Budget(), max_iterations=3, max_concurrent=2)
    final, seen, _ = run_graph(plan_obj(*t), verdicts=all_incomplete(t), budget=budget)

    assert len(seen) == len(set(seen)), f"attempt 重复：{seen}"
    per_task: dict[str, list[int]] = {}
    for tid, att in seen:
        per_task.setdefault(tid, []).append(att)
    for tid, atts in per_task.items():
        assert atts == sorted(set(atts)), f"{tid} 的 attempt 不是严格递增：{atts}"
    assert check_invariants(final, budget) == []


def test_dispatch_count_equals_recorded_attempts():
    """I2b：attempts 的记录值必须等于实际执行次数。"""
    t = [task("sq_001", priority=10), task("sq_002", priority=9), task("sq_003", priority=8)]
    budget = with_overrides(Budget(), max_concurrent=2)
    final, seen, _ = run_graph(plan_obj(*t), verdicts=all_incomplete(t), budget=budget)
    actual: dict[str, int] = {}
    for tid, _ in seen:
        actual[tid] = actual.get(tid, 0) + 1
    assert actual == dict(final["attempts"]), (actual, final["attempts"])
    assert check_invariants(final, budget) == []


# ── 规划清洗 ──────────────────────────────────────────────

def test_plan_without_criteria_is_rejected():
    bad = task("sq_001")
    bad["verification_criteria"] = ""
    good = task("sq_002", domain="metrics")
    final, seen, _ = run_graph(plan_obj(bad, good), verdicts=all_complete([good]))
    assert [tid for tid, _ in seen] == ["sq_002"], seen
    assert any("verification_criteria" in w for w in final["plan_warnings"])


def test_plan_with_unknown_domain_is_rejected():
    good = task("sq_001")
    bad = task("sq_002", domain="不存在的领域")
    final, seen, _ = run_graph(plan_obj(good, bad), verdicts=all_complete([good]))
    assert [tid for tid, _ in seen] == ["sq_001"], seen
    assert any("不在注册表" in w for w in final["plan_warnings"])


def test_dangling_dependency_is_removed():
    t = [task("sq_001", deps=["sq_999"])]
    final, seen, _ = run_graph(plan_obj(*t), verdicts=all_complete(t))
    assert seen == [("sq_001", 1)], seen
    assert any("不存在" in w for w in final["plan_warnings"])
    assert check_invariants(final, Budget()) == []


def test_cycle_is_broken_not_deadlocked():
    t = [task("sq_001", deps=["sq_002"]), task("sq_002", deps=["sq_001"])]
    final, seen, _ = run_graph(plan_obj(*t), verdicts=all_complete(t))
    assert sorted(tid for tid, _ in seen) == ["sq_001", "sq_002"], seen
    assert any("环路" in w for w in final["plan_warnings"])
    assert check_invariants(final, Budget()) == []


def test_duplicate_id_dropped():
    t1 = task("sq_001", priority=10)
    t2 = task("sq_001", priority=9)
    final, seen, _ = run_graph(plan_obj(t1, t2), verdicts=all_complete([t1]))
    assert seen == [("sq_001", 1)], seen
    assert any("重复" in w for w in final["plan_warnings"])


def test_max_tasks_truncation_drops_dependents_gracefully():
    tasks = [task(f"sq_{i:03d}", priority=10 - i) for i in range(1, 8)]   # 7 个，上限 6
    tasks[6]["dependencies"] = ["sq_001"]
    final, seen, _ = run_graph(plan_obj(*tasks), verdicts=all_complete(tasks))
    assert len(final["tasks"]) == 6
    assert any("上限" in w for w in final["plan_warnings"])
    assert check_invariants(final, Budget()) == []


# ── 失败与验证缺口 ─────────────────────────────────────────

def test_worker_failure_still_reaches_synthesis():
    t = [task("sq_001"), task("sq_002", domain="metrics")]
    final, seen, _ = run_graph(plan_obj(*t), verdicts=all_complete(t), fail_task_ids={"sq_001"})
    assert len(seen) >= 2
    assert final["final_answer"]
    failed = [r for r in final["results"] if not r["ok"]]
    assert failed and failed[0]["task_id"] == "sq_001"
    assert check_invariants(final, Budget()) == []


def test_verification_gap_does_not_silently_pass():
    t = [task("sq_001"), task("sq_002", domain="metrics")]
    partial = all_complete([t[0]])          # 故意漏掉 sq_002 的判定
    final, _, _ = run_graph(plan_obj(*t), verdicts=partial)
    v = {x["task_id"]: x for x in final["verdicts"]}
    assert v["sq_002"]["status"] == "incomplete"
    assert "验证器未对该结果给出结论" in v["sq_002"]["missing_aspects"][0]


# ── 双模型 ────────────────────────────────────────────────

def test_verify_and_replan_use_verify_slot():
    """验证与重规划必须走验证槽位，规划与综合走执行槽位。

    这条不能只看配置 —— 要断言实际调用落在哪个槽位上。
    """
    t = [task("sq_001")]
    budget = with_overrides(Budget(), max_iterations=3)
    _, _, fake = run_graph(plan_obj(*t), verdicts=all_incomplete(t), budget=budget)

    by_role: dict[str, set[str]] = {}
    for role, slot in fake.calls:
        by_role.setdefault(role, set()).add(slot)

    assert by_role["verify"] == {SLOT_VERIFY}, by_role
    assert by_role["replan"] == {SLOT_VERIFY}, by_role
    assert by_role["plan"] == {"exec"}, by_role
    assert by_role["synth"] == {"exec"}, by_role
