"""图状态 schema 与 reducer。

两条规则（都是上一轮实测踩出来的）：

1. **凡是被并行 worker 写入的通道，都必须声明 reducer。**
   没声明就是 last_value 语义，多个并行节点同时写会直接抛
   InvalidUpdateError: Can receive only one value per step。

2. **不要用布尔量表达"还有没有未验证结果"。**
   多个并行 worker 同时写一个布尔量同样会撞车；改用 `verified: task_id -> attempt`
   映射，既避开并发写，又能正确处理重试（新 attempt 大于已验证的 attempt 就意味着
   还要再验一次）。

状态分区（谁写哪一块，决定了哪些字段必须声明 reducer）：

OrchState
├── 输入区 (只读)
│   └── query: 全局原始输入
├── 规划产物区 (Plan 独占写，Last-Value)
│   ├── tasks: 子问题 DAG 列表
│   └── plan_warnings: 规划期清洗记录 (环路截断、丢弃不合规等)
├── 执行产物区 (Worker 并行写，必须带 Reducer)
│   ├── results: [SubResult] (merge_results 留最新 attempt)
│   ├── attempts: {task_id: int} (merge_int_max 记录派发次数)
│   ├── tokens_used: int (operator.add 累加总 Token)
│   ├── tokens_by_model: {model: int} (merge_token_maps 累加各模型 Token)
│   └── events: [dict] (append_list 记录并行时钟区间)
├── 验证产物区 (Verify 独占写)
│   ├── verdicts: [Verdict] (merge_verdicts 覆盖打分)
│   ├── verified: {task_id: int} (merge_int_max 记录已校验的最大 attempt)
│   ├── iteration: int (当前迭代轮数，用于收益递减判断)
│   ├── completeness: float (全局完成度均值，对比 0.8 阈值)
│   ├── retry_ids: [str] (触发重试的子任务 ID 列表)
│   └── stop_reason: str (触发停机的具体原因)
├── 监督拦截区 (Worker 并行写)
│   └── intercepts: [dict] (append_list 记录净化省掉的字符与防护动作)
├── 交付产物区 (Synth 独占写)
│   └── final_answer: str (最终综合结果)
└── 运行元信息 (调用方只读传入)
    ├── verify_is_independent: bool (验证模型与执行模型是否严格物理隔离)
    ├── exec_model: str (执行主力模型名)
    └── verify_model: str (质检验证模型名)
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class SubTask(TypedDict):
    """一个可独立执行的子问题。字段沿用 VMAO 表 1，agent_type 换成 domain。"""

    id: str
    question: str
    domain: str
    dependencies: list[str]
    priority: int
    context_from_deps: bool
    verification_criteria: str


class SubResult(TypedDict):
    """一个子问题的执行结果。结构化契约，不做散文式交接。"""

    task_id: str
    domain: str
    engine: str
    ok: bool
    text: str
    sources: list[str]
    input_tokens: int
    output_tokens: int
    error: str | None
    attempt: int
    purified: bool
    purged_chars: int


class Verdict(TypedDict):
    """验证结论。允许并强制要求指出缺失与矛盾。"""

    task_id: str
    status: str               # complete | partial | incomplete
    completeness: float       # 0..1
    missing_aspects: list[str]
    contradictions: list[str]
    recommendation: str       # accept | retry | escalate
    reason: str


class WorkerPayload(TypedDict):
    """Send 扇出时传给 worker 的载荷。

    只带当前子问题、原始任务、依赖结果 —— 绝不带其它 worker 的原文。
    """

    task: SubTask
    query: str
    dep_context: str
    attempt: int


# ── reducer ────────────────────────────────────────────────

def merge_results(existing: list[SubResult], new: list[SubResult]) -> list[SubResult]:
    """按 task_id 合并，保留 attempt 更大的那条。

    并行写入靠它收敛；重试也靠它做到 VMAO 的 result preservation
    （旧结果不被清空，只是被更高 attempt 覆盖）。
    """
    by_id: dict[str, SubResult] = {}
    for item in list(existing or []) + list(new or []):
        cur = by_id.get(item["task_id"])
        if cur is None or item.get("attempt", 0) >= cur.get("attempt", 0):
            by_id[item["task_id"]] = item
    return sorted(by_id.values(), key=lambda r: r["task_id"])


def merge_verdicts(existing: list[Verdict], new: list[Verdict]) -> list[Verdict]:
    by_id: dict[str, Verdict] = {}
    for item in list(existing or []) + list(new or []):
        by_id[item["task_id"]] = item
    return sorted(by_id.values(), key=lambda v: v["task_id"])


def merge_int_max(existing: dict[str, int], new: dict[str, int]) -> dict[str, int]:
    """取最大值。attempts 与 verified 都只应单调不减。"""
    out = dict(existing or {})
    for key, val in (new or {}).items():
        out[key] = max(out.get(key, 0), int(val))
    return out


def merge_token_maps(existing: dict[str, int], new: dict[str, int]) -> dict[str, int]:
    """按模型名累加 token。两个模型要分开计费。"""
    out = dict(existing or {})
    for key, val in (new or {}).items():
        out[key] = out.get(key, 0) + int(val)
    return out


def append_list(existing: list, new: list) -> list:
    return list(existing or []) + list(new or [])


# ── 状态 ──────────────────────────────────────────────────

class OrchState(TypedDict):
    query: str
    tasks: list[SubTask]
    results: Annotated[list[SubResult], merge_results]
    verdicts: Annotated[list[Verdict], merge_verdicts]
    attempts: Annotated[dict[str, int], merge_int_max]
    verified: Annotated[dict[str, int], merge_int_max]
    tokens_used: Annotated[int, operator.add]
    # 两个模型分别计费，混在一起算不出真实成本
    tokens_by_model: Annotated[dict[str, int], merge_token_maps]
    iteration: int
    completeness: float
    retry_ids: list[str]
    stop_reason: str
    plan_warnings: list[str]
    intercepts: Annotated[list[dict[str, Any]], append_list]
    events: Annotated[list[dict[str, Any]], append_list]
    final_answer: str
    verify_is_independent: bool
    exec_model: str
    verify_model: str


def initial_state(query: str, *, exec_model: str = "", verify_model: str = "",
                  verify_is_independent: bool = False) -> OrchState:
    return OrchState(
        query=query,
        tasks=[],
        results=[],
        verdicts=[],
        attempts={},
        verified={},
        tokens_used=0,
        tokens_by_model={},
        iteration=0,
        completeness=0.0,
        retry_ids=[],
        stop_reason="",
        plan_warnings=[],
        intercepts=[],
        events=[],
        final_answer="",
        verify_is_independent=verify_is_independent,
        exec_model=exec_model,
        verify_model=verify_model,
    )


def latest_results(state: OrchState) -> dict[str, SubResult]:
    return {r["task_id"]: r for r in state["results"]}


def unverified_results(state: OrchState) -> list[SubResult]:
    verified = state.get("verified") or {}
    return [r for r in state["results"] if verified.get(r["task_id"], 0) < r["attempt"]]
