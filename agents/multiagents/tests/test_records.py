"""落盘层与工具返回格式的测试 —— 全程零模型调用。

覆盖三件容易退化的事：
1. 记录必须落在 `.bear/multiagents/<session_id>/`，不同 session 目录隔离；
2. 编排抛异常时必须留下 `status=failed` 的记录，不能只有成功案例；
3. 工具返回文本里"可信度元信息"必须排在结论之前 —— 主 agent-loop 会
   `_persist_large_result` 截断（阈值 30KB，只保留前 200 行），结论放前面会把
   元信息整段截掉。
"""
from __future__ import annotations

import json

import pytest

from agents.multiagents import interface, records
from agents.multiagents.config import ModelConfig


# ── 打桩：把模型配置与跑图都换掉，测试不碰网络 ──────────────
@pytest.fixture
def stub_llm(monkeypatch):
    from agents.multiagents import llm

    stub = ModelConfig(api_key="k", api_base="https://example.invalid/v1",
                       exec_model="exec-test", verify_model="verify-test",
                       verify_is_independent=True, embedding_model="vec-test")
    monkeypatch.setattr(llm, "model_config", lambda: stub)
    monkeypatch.setattr(llm, "set_model_config", lambda cfg: None)
    return stub


def fake_record(*, answer: str = "这是结论", complete: bool = True) -> dict:
    return {
        "query": "测试任务",
        "started_at": "2026-10-07T04:00:00+00:00",
        "wall_seconds": 12.5,
        "models": {"exec": "exec-test", "verify": "verify-test",
                   "verify_is_independent": True},
        "summary": {"completeness": 1.0 if complete else 0.5, "iteration": 1,
                    "dispatches": 2, "tokens_used": 12345, "stop_reason": "达到完成度阈值"},
        "invariant_violations": [],
        "parallelism": {"spans": 2, "sum_seconds": 10.0, "union_seconds": 6.0,
                        "overlap_factor": 1.67},
        "cost_by_node": {"worker": 10000, "verify": 2000, "plan": 345},
        "cost_by_model": {"exec-test": 10345, "verify-test": 2000},
        "tasks": [{"id": "sq_001", "question": "q", "domain": "code", "dependencies": [],
                   "priority": 5, "context_from_deps": False, "verification_criteria": "c"}],
        "results": [{"task_id": "sq_001", "domain": "code", "engine": "lg-tool-agent",
                     "ok": True, "text": "r", "sources": [], "input_tokens": 10,
                     "output_tokens": 5, "error": None, "attempt": 1,
                     "purified": False, "purged_chars": 0}],
        "verdicts": [{"task_id": "sq_001", "status": "complete" if complete else "partial",
                      "completeness": 1.0 if complete else 0.5,
                      "missing_aspects": [] if complete else ["缺少逐文件行数"],
                      "contradictions": [] if complete else ["行数口径不一致 10296 vs 10273"],
                      "recommendation": "accept" if complete else "retry", "reason": "r"}],
        "intercepts": [],
        "events": [{"node": "plan"}],
        "plan_warnings": [],
        "final_answer": answer,
        "stop_reason": "达到完成度阈值",
    }


# ── records.py ────────────────────────────────────────────
def test_session_dir_isolated_by_session(tmp_path):
    a = records.session_dir("aaaaaaaa", root=tmp_path)
    b = records.session_dir("bbbbbbbb", root=tmp_path)
    assert a != b
    assert a.is_dir() and b.is_dir()
    assert a == tmp_path / "aaaaaaaa"


def test_manual_session_when_id_missing(tmp_path):
    """没有 session_id 时退化到 _manual，不静默丢记录。"""
    d = records.session_dir(None, root=tmp_path)
    assert d.name == records.MANUAL_SESSION
    assert d.is_dir()


def test_save_run_writes_json_and_index(tmp_path):
    path = records.save_run(fake_record(), session_id="sess0001", root=tmp_path)

    assert path.is_file()
    assert path.parent == tmp_path / "sess0001"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["session_id"] == "sess0001"
    assert payload["status"] == "ok"

    index = records.load_index("sess0001", root=tmp_path)
    assert len(index) == 1
    assert index[0]["record"] == path.name
    assert index[0]["status"] == "ok"
    assert index[0]["tokens"] == 12345


def test_save_run_twice_in_same_second_does_not_collide(tmp_path):
    p1 = records.save_run(fake_record(), session_id="s", root=tmp_path)
    p2 = records.save_run(fake_record(), session_id="s", root=tmp_path)
    assert p1 != p2
    assert len(records.load_index("s", root=tmp_path)) == 2


def test_atomic_write_leaves_no_tmp(tmp_path):
    records.save_run(fake_record(), session_id="s", root=tmp_path)
    leftovers = list((tmp_path / "s").glob("*.tmp"))
    assert leftovers == [], f"残留临时文件：{leftovers}"


def test_save_failure_records_status_and_traceback(tmp_path):
    try:
        raise ValueError("模型返回的不是 JSON")
    except ValueError as exc:
        path = records.save_failure(session_id="s", task="t", error=exc,
                                    wall_seconds=3.0, root=tmp_path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert "ValueError" in payload["error"]
    assert "Traceback" in payload["traceback"]

    index = records.load_index("s", root=tmp_path)
    assert index[-1]["status"] == "failed"
    assert "ValueError" in index[-1]["error"]


def test_summarise_keeps_no_body_text(tmp_path):
    """索引行必须短：不能把结论正文塞进 runs.jsonl。"""
    entry = records.summarise(fake_record(answer="X" * 5000))
    assert "X" * 100 not in json.dumps(entry, ensure_ascii=False)
    assert entry["contradictions"] == 0
    assert entry["unfinished"] == 0


def test_summarise_counts_unfinished_and_contradictions():
    entry = records.summarise(fake_record(complete=False))
    assert entry["unfinished"] == 1
    assert entry["contradictions"] == 1


# ── interface.py ──────────────────────────────────────────
async def _ok_run(query, *, budget):
    return fake_record()


def test_execute_multiagent_writes_into_session_dir(tmp_path, stub_llm, monkeypatch):
    monkeypatch.setattr(interface, "run_orchestration", _ok_run)

    import asyncio
    text = asyncio.run(interface.execute_multiagent(
        {"task": "测试任务"}, session_id="sess0002", records_root_override=tmp_path))

    files = sorted(p.name for p in (tmp_path / "sess0002").glob("*"))
    assert any(f.startswith("run-") and f.endswith(".json") for f in files), files
    assert records.INDEX_FILENAME in files
    assert "【结论】" in text
    assert "这是结论" in text


def test_empty_task_is_rejected_without_touching_disk(tmp_path, stub_llm):
    import asyncio
    text = asyncio.run(interface.execute_multiagent(
        {"task": "   "}, session_id="s", records_root_override=tmp_path))
    assert text.startswith("Error:")
    assert not (tmp_path / "s").exists(), "参数非法时不该建目录"


def test_bad_max_iterations_is_rejected(tmp_path, stub_llm):
    import asyncio
    text = asyncio.run(interface.execute_multiagent(
        {"task": "t", "max_iterations": "很多"}, session_id="s",
        records_root_override=tmp_path))
    assert text.startswith("Error:")
    assert not (tmp_path / "s").exists()


def test_failure_leaves_a_record_and_propagates(tmp_path, stub_llm, monkeypatch):
    async def _boom(query, *, budget):
        raise RuntimeError("端点不可用")

    monkeypatch.setattr(interface, "run_orchestration", _boom)

    import asyncio
    with pytest.raises(RuntimeError, match="端点不可用"):
        asyncio.run(interface.execute_multiagent(
            {"task": "t"}, session_id="sess0003", records_root_override=tmp_path))

    index = records.load_index("sess0003", root=tmp_path)
    assert len(index) == 1 and index[0]["status"] == "failed"
    assert "RuntimeError" in index[0]["error"]


# ── 返回文本的顺序（回归测试）─────────────────────────────
def test_meta_block_precedes_answer():
    text = interface.format_tool_result(fake_record(answer="AAAA"))
    assert text.index("编排结果元信息") < text.index("【结论】")
    assert text.index("【结论】") < text.index("AAAA")


def test_meta_survives_first_200_line_truncation():
    """模拟 Agent._persist_large_result：超 30KB 只保留前 200 行。

    结论放前面时，元信息会在这一步被整段截掉；这条测试锁住"元信息在前"。
    """
    answer = "\n".join(f"第 {i} 行结论" for i in range(400))
    text = interface.format_tool_result(fake_record(answer=answer, complete=False))

    preview = "\n".join(text.split("\n")[:200])
    assert "编排结果元信息" in preview, "元信息被截断了 —— 顺序又被打回结论在前"
    assert "未完成或部分完成的子问题" in preview
    assert "验证发现的矛盾" in preview


def test_independent_verify_flag_is_surfaced():
    """验证与执行同源时必须写进返回文本，不能只留在配置里。"""
    rec = fake_record()
    rec["models"]["verify_is_independent"] = False
    assert "共享偏差" in interface.format_tool_result(rec)


# ── token 拆分（供宿主 Agent 精确记账）─────────────────────
def _usage_record(*, worker_in: int, worker_out: int, non_worker_total: int,
                  reported: int | None = None) -> dict:
    """一个 lg-tool worker（in/out 分开）+ 非 worker 节点（只有一个总数）。"""
    rec = fake_record()
    rec["results"][0]["engine"] = "lg-tool-agent"
    rec["results"][0]["input_tokens"] = worker_in
    rec["results"][0]["output_tokens"] = worker_out
    rec["summary"]["tokens_used"] = (
        worker_in + worker_out + non_worker_total if reported is None else reported)
    return rec


def test_usage_splits_input_output_exactly(monkeypatch):
    """llm 计数器给非 worker 部分，SubResult 给 lg-tool worker 部分，两份相加。"""
    from agents.multiagents import llm

    monkeypatch.setattr(llm, "usage_snapshot",
                        lambda: {"input": 700, "output": 300})   # plan/verify/synth + plain worker
    rec = _usage_record(worker_in=5000, worker_out=1000, non_worker_total=1000)

    usage = interface._usage_from(rec)
    assert usage["input"] == 5700        # 700 + 5000
    assert usage["output"] == 1300       # 300 + 1000
    assert usage["total"] == 7000
    assert usage["reported_total"] == 7000
    assert usage["accounting_mismatch"] == 0


def test_plain_worker_tokens_are_not_double_counted(monkeypatch):
    """plain-llm worker 的 token 已经在 llm 计数器里，不能再从 SubResult 加一遍。"""
    from agents.multiagents import llm

    monkeypatch.setattr(llm, "usage_snapshot", lambda: {"input": 100, "output": 50})
    rec = fake_record()
    rec["results"][0]["engine"] = "plain-llm"        # 不走 lg-tool 分支
    rec["results"][0]["input_tokens"] = 999999       # 故意给一个会被双计的值
    rec["results"][0]["output_tokens"] = 999999
    rec["summary"]["tokens_used"] = 150

    usage = interface._usage_from(rec)
    assert usage["input"] == 100 and usage["output"] == 50
    assert usage["accounting_mismatch"] == 0


def test_accounting_mismatch_is_reported_not_swallowed(monkeypatch):
    """两份之和与事件合计对不上时必须写出来，不能静默。"""
    from agents.multiagents import llm

    monkeypatch.setattr(llm, "usage_snapshot", lambda: {"input": 700, "output": 300})
    rec = _usage_record(worker_in=5000, worker_out=1000, non_worker_total=1000,
                        reported=9999)                       # 事件说 9999，实际算出来 7000

    usage = interface._usage_from(rec)
    assert usage["accounting_mismatch"] == 7000 - 9999

    rec["accounting_mismatch"] = usage["accounting_mismatch"]
    assert "token 记账不一致" in interface.format_tool_result(rec)


def test_usage_is_empty_when_arguments_invalid(tmp_path, stub_llm):
    import asyncio
    text, usage = asyncio.run(interface.execute_multiagent_with_usage(
        {"task": ""}, session_id="s", records_root_override=tmp_path))
    assert text.startswith("Error:")
    assert usage == {}


def test_with_usage_returns_usage_for_a_real_run(tmp_path, stub_llm, monkeypatch):
    from agents.multiagents import llm

    async def _ok(query, *, budget):
        return _usage_record(worker_in=5000, worker_out=1000, non_worker_total=1000)

    monkeypatch.setattr(interface, "run_orchestration", _ok)
    monkeypatch.setattr(llm, "usage_snapshot", lambda: {"input": 700, "output": 300})

    import asyncio
    text, usage = asyncio.run(interface.execute_multiagent_with_usage(
        {"task": "t"}, session_id="sess-usage", records_root_override=tmp_path))

    assert usage["total"] == 7000
    assert usage["input"] == 5700
    assert "【结论】" in text


def test_prepare_resets_usage_counter_before_running(tmp_path, stub_llm, monkeypatch):
    """每次运行前必须清零，否则第二次运行会把第一次的用量算进来。"""
    from agents.multiagents import llm

    calls: list[str] = []
    monkeypatch.setattr(llm, "reset_usage", lambda: calls.append("reset"))

    async def _ok(query, *, budget):
        calls.append("run")
        return fake_record()

    monkeypatch.setattr(interface, "run_orchestration", _ok)

    import asyncio
    asyncio.run(interface.execute_multiagent_with_usage(
        {"task": "t"}, session_id="s", records_root_override=tmp_path))
    assert calls == ["reset", "run"], calls
