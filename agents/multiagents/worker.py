"""Worker 执行引擎 —— 编排层与执行层的边界。

两个引擎共用同一个结果契约（SubResult）：
  lg-tool-agent   create_agent 驱动，工具转调 agents/tools.py（默认）
  plain-llm       单次模型调用，无工具

每个 worker 自带一个 TaskSupervisor，挂在工具通道上。监督状态的收口在 run_worker
的返回值里，图把它写进 state.intercepts。

注意：不用项目自己的 Agent 类（agents/agent.py）当引擎。它有一条独立缺陷：
_chat_openai 里 oai_batches 的构建与消费写在了 `for tc in tool_calls` 循环内部
（L1907-L1959），模型一轮发多个工具调用时早先结果会被重复追加，接口直接报
"Duplicate value for 'tool_call_id'"。这属于另一个问题，不混进本方案。
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Awaitable, Callable

from .config import DEFAULT_BUDGET, Budget
from .domains import ENGINE_LG_TOOL, ENGINE_PLAIN, Domain
from .llm import OrchestratorError, call_json, extract_json, is_rate_limit_error, model_config
from .state import SubResult, SubTask
from .supervisor import TaskSupervisor
from .tools import TOOL_REGISTRY, wrap_with_supervisor

FORBIDDEN_FOR_WORKERS = {
    "agent", "write_file", "edit_file", "run_shell", "skill", "skill_create",
    "skill_evolve", "tool_search", "compact_context", "enter_plan_mode", "exit_plan_mode",
}


def _failure(task: SubTask, domain: Domain, attempt: int, error: str) -> SubResult:
    return SubResult(
        task_id=task["id"], domain=domain.name, engine=domain.engine, ok=False, text="",
        sources=[], input_tokens=0, output_tokens=0, error=error, attempt=attempt,
        purified=False, purged_chars=0,
    )


def _package(task: SubTask, domain: Domain, engine: str, attempt: int, text: str,
             in_tokens: int, out_tokens: int, supervisor: TaskSupervisor) -> SubResult:
    """把 worker 原始输出收敛成 SubResult。JSON 解析失败视为失败，不当成功。"""
    purged = supervisor.purged_total
    try:
        payload = extract_json(text)
    except OrchestratorError as exc:
        res = _failure(task, domain, attempt, f"worker 输出不是合法 JSON：{exc}")
        res.update(engine=engine, text=text, input_tokens=in_tokens, output_tokens=out_tokens,
                   purified=purged > 0, purged_chars=purged)
        return res

    if not isinstance(payload, dict):
        res = _failure(task, domain, attempt, "worker 输出的 JSON 不是对象")
        res.update(engine=engine, input_tokens=in_tokens, output_tokens=out_tokens)
        return res

    answer = str(payload.get("answer") or "").strip()
    gaps = payload.get("gaps") or []
    ok = bool(answer)
    return SubResult(
        task_id=task["id"], domain=domain.name, engine=engine, ok=ok, text=answer,
        sources=[str(s) for s in (payload.get("sources") or [])][:12],
        input_tokens=in_tokens, output_tokens=out_tokens,
        error=None if ok else (f"worker 未给出可用结论；gaps={gaps}" if gaps else "worker 未给出可用结论"),
        attempt=attempt, purified=purged > 0, purged_chars=purged,
    )


def build_task_prompt(task: SubTask, query: str, dep_context: str) -> str:
    """子问题的 prompt 固定三段：总体任务 / 你负责的子问题 / 完成判据。

    worker 只能看到这三段（加依赖结果），看不到全局对话历史 ——
    这是 MAST FC1.1 与 FC2.3 的对策。
    """
    parts = [
        f"# 总体任务\n{query}",
        f"# 你负责的子问题\n{task['question']}",
        f"# 完成判据\n{task['verification_criteria']}",
    ]
    if dep_context:
        parts.append(f"# 上游依赖的结果（只信这些，不要另行推测）\n{dep_context}")
    parts.append("只回答你负责的子问题，不要替其它领域作答。")
    return "\n\n".join(parts)


# ── 引擎一：create_agent + 项目工具（默认）──────────────────
async def _engine_lg_tool(task: SubTask, domain: Domain, query: str, dep_context: str,
                          attempt: int, supervisor: TaskSupervisor, budget: Budget) -> SubResult:
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI

    missing = sorted(set(domain.tools) - set(TOOL_REGISTRY))
    if missing:
        raise OrchestratorError(f"领域 {domain.name} 声明的工具未注册：{missing}")

    cfg = model_config()
    model = ChatOpenAI(model=cfg.exec_model, api_key=cfg.api_key, base_url=cfg.openai_base,
                       temperature=0.0, max_retries=budget.llm_max_retries,
                       timeout=budget.llm_timeout_s)
    tools = [
        wrap_with_supervisor(TOOL_REGISTRY[name], supervisor)
        for name in domain.tools
        if name not in FORBIDDEN_FOR_WORKERS
    ]
    agent = create_agent(model, tools=tools, system_prompt=domain.system_prompt,
                         name=f"{domain.name}_worker")
    out = await agent.ainvoke(
        {"messages": [{"role": "user", "content": build_task_prompt(task, query, dep_context)}]},
        config={"recursion_limit": budget.worker_recursion_limit},
    )

    messages = out.get("messages") or []
    text = ""
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if isinstance(content, str) and content.strip():
            text = content
            break
    in_tok = out_tok = 0
    for msg in messages:
        usage = getattr(msg, "usage_metadata", None)
        if usage:
            in_tok += int(usage.get("input_tokens", 0) or 0)
            out_tok += int(usage.get("output_tokens", 0) or 0)
    return _package(task, domain, ENGINE_LG_TOOL, attempt, text, in_tok, out_tok, supervisor)


# ── 引擎二：纯模型调用 ────────────────────────────────────
async def _engine_plain(task: SubTask, domain: Domain, query: str, dep_context: str,
                        attempt: int, supervisor: TaskSupervisor, budget: Budget) -> SubResult:
    payload, tokens = await asyncio.to_thread(
        call_json, domain.system_prompt, build_task_prompt(task, query, dep_context))
    text = json.dumps(payload, ensure_ascii=False) if isinstance(payload, dict) else str(payload)
    return _package(task, domain, ENGINE_PLAIN, attempt, text,
                    tokens["input"], tokens["output"], supervisor)


_ENGINES: dict[str, Callable[..., Awaitable[SubResult]]] = {
    ENGINE_LG_TOOL: _engine_lg_tool,
    ENGINE_PLAIN: _engine_plain,
}

# 测试注入点：把执行整个替掉，可以零 token 跑通全图
_engine_override: Callable[..., Awaitable[SubResult]] | None = None


def set_engine_override(fn: Callable[..., Awaitable[SubResult]] | None) -> None:
    global _engine_override
    _engine_override = fn


async def run_worker(task: SubTask, domain: Domain, query: str, dep_context: str, *,
                     attempt: int = 1, budget: Budget = DEFAULT_BUDGET,
                     ) -> tuple[SubResult, list[dict]]:
    """执行一个子任务。返回 (结果, 监督记录)。

    限流（429）单独重试：它是外部端点的瞬时状态，让它变成 ok=False 再走重规划
    既慢又贵。其它异常一律收敛成 ok=False 交给验证环节，不静默吞掉。
    """
    supervisor = TaskSupervisor(task_id=task["id"], budget=budget)
    started = time.monotonic()

    engine = _engine_override or _ENGINES.get(domain.engine)
    if engine is None:
        return _failure(task, domain, attempt, f"未知引擎 {domain.engine!r}"), []

    for rl_try in range(3 + 1):
        try:
            result = await asyncio.wait_for(
                engine(task, domain, query, dep_context, attempt, supervisor, budget),
                timeout=budget.agent_timeout_s)
            return result, supervisor.as_dicts()
        except asyncio.TimeoutError:
            return _failure(task, domain, attempt, f"超时（>{budget.agent_timeout_s}s）"), supervisor.as_dicts()
        except Exception as exc:  # noqa: BLE001 —— 必须收住，否则单点失败会炸掉整个扇出
            if is_rate_limit_error(exc) and rl_try < 3:
                await asyncio.sleep(8.0 * (rl_try + 1))
                continue
            elapsed = time.monotonic() - started
            return (_failure(task, domain, attempt,
                             f"{type(exc).__name__}: {exc}（耗时 {elapsed:.1f}s）"),
                    supervisor.as_dicts())

    return _failure(task, domain, attempt, "限流重试 3 次仍失败"), supervisor.as_dicts()
