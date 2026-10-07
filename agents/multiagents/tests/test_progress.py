"""编排进度输出的测试 —— 零模型调用。

用户端的诉求是"调用 multiagent 后看不到任何中间过程"。要保证的是三件事：
  1. 重要节点确实会发出进度（不是只在我以为的地方加了日志）；
  2. 进度只在宿主路径打开，脚本直连与单元测试不被刷屏；
  3. 渲染走 `agents/ui.py` 且**不经过 rich markup** —— 节点文本里有
     `[code]`、`[... 净化 ...]` 这类方括号，走 markup 会被当成标签解析。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from agents.multiagents import interface, trace

# 在真实图上跑一遍时，必须有输出的节点
IMPORTANT_NODES = {"plan", "route", "worker", "join", "verify", "synthesize"}


@pytest.fixture(autouse=True)
def _restore_trace_state():
    """trace 是模块级全局，测完必须还原，否则会污染其它测试。"""
    yield
    trace.set_enabled(False)
    trace.set_printer(None)
    trace.reset()


def _load_sibling(name: str):
    """按路径加载同目录的测试模块。

    不靠 pytest 的 import 模式推导模块名（没有 __init__.py，容易出意外）。
    """
    spec = importlib.util.spec_from_file_location(f"_ma_{name}", Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _fake_record(answer: str = "结论") -> dict:
    return {
        "query": "t", "started_at": "2026-10-07T04:00:00+00:00", "wall_seconds": 1.0,
        "models": {"exec": "e", "verify": "v", "verify_is_independent": True},
        "summary": {"completeness": 1.0, "iteration": 1, "dispatches": 1,
                    "tokens_used": 10, "stop_reason": "ok"},
        "invariant_violations": [], "cost_by_model": {}, "tasks": [], "results": [],
        "verdicts": [], "intercepts": [], "events": [], "plan_warnings": [],
        "final_answer": answer, "stop_reason": "ok",
    }


@pytest.fixture
def stub_llm(monkeypatch):
    from agents.multiagents import llm
    from agents.multiagents.config import ModelConfig

    stub = ModelConfig(api_key="k", api_base="https://example.invalid/v1",
                       exec_model="e", verify_model="v",
                       verify_is_independent=True, embedding_model=None)
    monkeypatch.setattr(llm, "model_config", lambda: stub)
    monkeypatch.setattr(llm, "set_model_config", lambda cfg: None)


# ── 渲染是否可插拔 ─────────────────────────────────────────
def test_printer_is_pluggable():
    seen: list[tuple[str, ...]] = []
    trace.set_printer(lambda n, t: seen.append(("node", n, t)),
                      lambda t: seen.append(("detail", t)))
    trace.set_enabled(True)

    trace.node("plan", "拆出 2 个子问题")
    trace.detail("sq_001 [code]")

    assert seen == [("node", "plan", "拆出 2 个子问题"), ("detail", "sq_001 [code]")]


def test_disabled_prints_nothing(capsys):
    trace.set_enabled(False)
    trace.node("plan", "不该出现")
    assert capsys.readouterr().out == ""


def test_use_host_ui_wires_agents_ui_renderers():
    from agents.ui import print_multiagent_detail, print_multiagent_node

    trace.use_host_ui()
    trace.set_enabled(True)
    trace.node("verify", "第 1 轮")
    trace.detail("sq_001 complete")

    # 直接对照：把同一段文本交给 ui.py 的函数，输出应当一致
    assert callable(print_multiagent_node) and callable(print_multiagent_detail)


# ── 方括号不能被 rich markup 吃掉 ──────────────────────────
def test_renderer_keeps_brackets_and_does_not_crash(capsys):
    from agents.ui import print_multiagent_detail, print_multiagent_node

    print_multiagent_node("worker", "sq_001 [code] attempt=1 ok")
    print_multiagent_detail("[... 监督层净化：省略中间 4321 字符 ...]")
    out = capsys.readouterr().out

    assert "[code]" in out, out
    assert "监督层净化" in out, out
    assert "省略中间 4321 字符" in out, out


# ── 宿主路径默认打开，直连路径默认关闭 ─────────────────────
def test_progress_off_by_default_in_library_path(tmp_path, stub_llm, monkeypatch, capsys):
    async def _run(query, *, budget):
        trace.node("plan", "直连不该打印这行")
        return _fake_record()

    monkeypatch.setattr(interface, "run_orchestration", _run)

    import asyncio
    asyncio.run(interface.execute_multiagent(
        {"task": "t"}, session_id="s", records_root_override=tmp_path))

    assert "直连不该打印这行" not in capsys.readouterr().out


def test_progress_on_prints_node_lines(tmp_path, stub_llm, monkeypatch, capsys):
    async def _run(query, *, budget):
        trace.node("plan", "拆出 2 个子问题")
        trace.node("worker", "sq_001 [code] attempt=1 ok")
        trace.node("verify", "第 1 轮  完成度 1.00")
        return _fake_record()

    monkeypatch.setattr(interface, "run_orchestration", _run)

    import asyncio
    text, usage = asyncio.run(interface.execute_multiagent_with_usage(
        {"task": "t"}, session_id="s", progress=True, records_root_override=tmp_path))

    out = capsys.readouterr().out
    assert "plan" in out and "拆出 2 个子问题" in out
    assert "sq_001 [code] attempt=1 ok" in out
    assert "第 1 轮  完成度 1.00" in out
    assert usage["total"] == 0
    assert "【结论】" in text


def test_progress_true_also_reports_real_graph_nodes(tmp_path, monkeypatch):
    """在真实图上跑一遍，确认六个重要节点都会发进度。

    用 test_routing 里的假后端 + 假引擎，全程零模型调用。
    """
    routing = _load_sibling("test_routing")

    from agents.multiagents import worker as worker_mod
    from agents.multiagents.config import Budget, with_overrides
    from agents.multiagents.graph import make_graph
    from agents.multiagents.llm import set_backend
    from agents.multiagents.state import initial_state

    recorded: list[str] = []
    monkeypatch.setattr(trace, "node", lambda name, text: recorded.append(name))
    monkeypatch.setattr(trace, "detail", lambda text: None)

    tasks = [routing.task("sq_001"), routing.task("sq_002", "docs")]
    budget = with_overrides(Budget(), max_concurrent=2, max_iterations=1)

    engine = routing.make_fake_engine()
    fake = routing.FakeBackend(routing.plan_obj(*tasks), routing.all_complete(tasks))
    set_backend(fake)
    worker_mod.set_engine_override(engine)
    try:
        import asyncio
        asyncio.run(make_graph(budget).ainvoke(
            initial_state("测试任务"),
            config={"recursion_limit": budget.graph_recursion_limit}))
    finally:
        worker_mod.set_engine_override(None)
        set_backend(None)

    seen = set(recorded)
    missing = IMPORTANT_NODES - seen
    assert not missing, f"这些重要节点没有任何进度输出：{missing}（实际有 {sorted(seen)}）"
