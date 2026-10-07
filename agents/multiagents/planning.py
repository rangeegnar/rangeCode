"""规划：把 query 拆成跨领域的子问题 DAG。

prompt 骨架来自 VMAO 附录 A 的 QueryPlanner（RAG First / Maximize Parallelism /
Minimize Dependencies / Be Specific），agent_type 换成 domain，加了两条：
跨领域必拆、判据必填。

本模块同时是 MAST FC1.1（违反任务规范，11.8%）与 FC1.2（违反角色规范）的守门人。
校验与清洗全部是纯函数 validate_plan()，可以离线单测。
"""
from __future__ import annotations

import json
from typing import Any

from .config import Budget
from .domains import DOMAINS, domain_catalog
from .llm import OrchestratorError, call_json
from .state import SubTask

PLANNER_SYSTEM = """你是一个规划器，负责把复杂任务拆解成可以由不同领域的专家并行执行的子问题。

可用领域：
{catalog}

拆解规则：
1. 跨领域必拆：如果任务的某个部分需要另一种能力，把它拆成独立子问题交给对应领域。
   目标是让尽可能多的子问题落在不同领域上，从而可以并行。
2. 最大化并行：没有依赖关系的子问题应该排在同一个波次。
3. 依赖最小化：只有当一个子问题确实需要另一个的输出作为输入时，才写依赖。
4. 判据必填：每个子问题的 verification_criteria 必须写成一句可判定的标准，
   说明"达到什么程度算回答了"。写不出判据的子问题说明它本身不够具体，请重新表述。
5. 具体化：每个子问题要清楚、可独立回答，不要出现"等等"、"相关分析"这类模糊表述。
6. 子问题数量控制在 {max_tasks} 个以内。宁可少拆也不要拆出无用的子问题。
7. 不要在子问题正文里引用其它子问题的编号（如"基于 sq_001 的数据"）；
   需要上游结果就写进 dependencies，运行时会自动注入。

只输出 JSON 对象：
{{
  "sub_questions": [
    {{
      "id": "sq_001",
      "question": "具体子问题",
      "domain": "上面列出的领域名之一",
      "dependencies": ["必须先完成的子问题 id"],
      "priority": 1到10的整数,
      "context_from_deps": true或false,
      "verification_criteria": "可判定的完成标准"
    }}
  ],
  "explanation": "为什么这样拆"
}}
"""


def validate_plan(raw: dict[str, Any], budget: Budget, *, max_tasks: int = 6) -> tuple[list[SubTask], list[str]]:
    """校验并清洗规划结果。所有丢弃与修补动作都留下可见的 warning。"""
    warnings: list[str] = []
    items = raw.get("sub_questions")
    if not isinstance(items, list) or not items:
        raise OrchestratorError(
            f"规划器没有产出任何 sub_questions：{json.dumps(raw, ensure_ascii=False)[:400]}")

    tasks: list[SubTask] = []
    seen: set[str] = set()

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            warnings.append(f"第 {idx + 1} 项不是对象，已丢弃")
            continue

        tid = str(item.get("id") or "").strip() or f"sq_{len(tasks) + 1:03d}"
        if tid in seen:
            warnings.append(f"id 重复：{tid}，已丢弃后出现的那条")
            continue

        question = str(item.get("question") or "").strip()
        criteria = str(item.get("verification_criteria") or "").strip()
        domain = str(item.get("domain") or "").strip()

        if not question:
            warnings.append(f"{tid} 缺 question，已丢弃")
            continue
        if not criteria:
            warnings.append(f"{tid} 缺 verification_criteria，已丢弃（不可验证的子问题不进入执行）")
            continue
        if domain not in DOMAINS:
            warnings.append(f"{tid} 的领域 {domain!r} 不在注册表里，已丢弃")
            continue

        try:
            priority = int(item.get("priority", 5))
        except (TypeError, ValueError):
            priority = 5
            warnings.append(f"{tid} 的 priority 不是整数，回落到 5")
        priority = min(10, max(1, priority))

        deps = item.get("dependencies") or []
        if not isinstance(deps, list):
            deps = []
            warnings.append(f"{tid} 的 dependencies 不是数组，按空处理")
        deps = [str(d).strip() for d in deps if str(d).strip() and str(d).strip() != tid]

        tasks.append(SubTask(
            id=tid, question=question, domain=domain, dependencies=deps,
            priority=priority, context_from_deps=bool(item.get("context_from_deps", False)),
            verification_criteria=criteria,
        ))
        seen.add(tid)

    if not tasks:
        raise OrchestratorError("规划结果校验后没有任何合法子问题。警告：" + "；".join(warnings))

    if len(tasks) > max_tasks:
        warnings.append(f"子问题数 {len(tasks)} 超过上限 {max_tasks}，按优先级截断")
        tasks.sort(key=lambda t: (-t["priority"], t["id"]))
        kept = {t["id"] for t in tasks[:max_tasks]}
        for t in tasks[max_tasks:]:
            warnings.append(f"{t['id']} 因超出上限被丢弃")
        tasks = tasks[:max_tasks]
        for t in tasks:
            dropped = [d for d in t["dependencies"] if d not in kept]
            if dropped:
                warnings.append(f"{t['id']} 依赖了被截断的 {dropped}，已摘除")
                t["dependencies"] = [d for d in t["dependencies"] if d in kept]

    # 悬空依赖：指向不存在的 id，调度时永远无法就绪
    ids = {t["id"] for t in tasks}
    for t in tasks:
        dangling = [d for d in t["dependencies"] if d not in ids]
        if dangling:
            warnings.append(f"{t['id']} 依赖了不存在的 {dangling}，已摘除")
            t["dependencies"] = [d for d in t["dependencies"] if d in ids]

    # 环路：有环就永远调度不出来，摘边而不是丢任务
    cycle_edges = find_cycle_edges(tasks)
    if cycle_edges:
        warnings.append(f"依赖存在环路，已摘除这些边：{sorted(cycle_edges)}")
        for t in tasks:
            t["dependencies"] = [d for d in t["dependencies"] if (t["id"], d) not in cycle_edges]

    return tasks, warnings


def find_cycle_edges(tasks: list[SubTask]) -> set[tuple[str, str]]:
    """找出构成环的边。摘掉这些边就能打破环。"""
    graph = {t["id"]: list(t["dependencies"]) for t in tasks}
    WHITE, GREY, BLACK = 0, 1, 2
    color = {tid: WHITE for tid in graph}
    bad: set[tuple[str, str]] = set()

    def dfs(node: str, stack: list[str]) -> None:
        color[node] = GREY
        stack.append(node)
        for dep in graph.get(node, []):
            if dep not in color:
                continue
            if color[dep] == GREY:
                start = stack.index(dep) if dep in stack else 0
                for a, b in zip(stack[start:], stack[start + 1:]):
                    bad.add((a, b))
                bad.add((node, dep))
            elif color[dep] == WHITE:
                dfs(dep, stack)
        stack.pop()
        color[node] = BLACK

    for tid in graph:
        if color[tid] == WHITE:
            dfs(tid, [])
    return bad


def plan(query: str, *, budget: Budget, max_tasks: int = 6) -> tuple[list[SubTask], list[str], int]:
    """返回 (子任务, 警告, 消耗 token)。走执行模型。"""
    system = PLANNER_SYSTEM.format(catalog=domain_catalog(), max_tasks=max_tasks)
    payload, tokens = call_json(system, f"需要拆解的任务：\n{query}")
    if not isinstance(payload, dict):
        raise OrchestratorError("规划器返回的不是 JSON 对象")
    tasks, warnings = validate_plan(payload, budget, max_tasks=max_tasks)
    return tasks, warnings, tokens["input"] + tokens["output"]
