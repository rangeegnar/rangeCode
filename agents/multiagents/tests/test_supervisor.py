"""监督层的测试 —— 全部是纯函数，零模型调用。

监督层的价值主张是"省 token 且不伤质量"，但过滤器与净化都是确定性逻辑，
所以正确性可以完全离线验证，只有"净效果"才需要真实运行。
"""
from __future__ import annotations

from agents.multiagents.config import Budget, with_overrides
from agents.multiagents.supervisor import (ACT_APPROVE, ACT_PURIFY, SIG_EXCESSIVE, SIG_LOOP, SIG_STEP_CHECK,
                           TaskSupervisor, is_excessive, purify, tool_signature)

B = Budget()


def obs(n: int, prefix: str = "line") -> str:
    """造一段带行号的观测，模拟 agents/tools.py 的 read_file 输出格式。"""
    return "\n".join(f"{i:5d} | {prefix} {i} " + "x" * 60 for i in range(1, n + 1))


def test_short_observation_untouched():
    text = "short"
    out, purged = purify(text, B)
    assert out == text and purged == 0
    assert not is_excessive(text, B)


def test_long_observation_is_purified():
    text = obs(300)                      # 约 20KB，远超 tau_len=3000
    assert is_excessive(text, B)
    out, purged = purify(text, B)
    assert purged > 0
    assert len(out) < len(text)
    assert "监督层净化" in out


def test_purify_keeps_line_numbers():
    """论文的「噪声即信号」：对代码阅读来说行号就是信号，净化不能把行号列整个丢掉。"""
    text = obs(300)
    out, _ = purify(text, B)
    head_lines = [ln for ln in out.split("\n")[:5] if ln.strip()]
    assert any("|" in ln and ln.strip().split("|")[0].strip().isdigit() for ln in head_lines), head_lines


def test_purify_is_deterministic():
    text = obs(300)
    a, pa = purify(text, B)
    b, pb = purify(text, B)
    assert a == b and pa == pb


def test_purify_head_tail_respects_threshold():
    text = obs(300)
    out, purged = purify(text, B)
    assert len(out) <= B.tau_len + 1, (len(out), B.tau_len)
    assert purged > 0


def test_purify_collapses_blank_runs_via_cheap_path():
    """空行成串时压空行就够落到阈值内 —— 走便宜路径，不做头尾截断。"""
    text = "a" * 100 + "\n" * 40 + "b" * 100          # 240 字符
    budget = with_overrides(B, tau_len=220)
    out, purged = purify(text, budget)
    assert "\n\n\n" not in out
    assert purged == 38
    assert "监督层净化" not in out


def test_purify_gives_up_when_threshold_too_small():
    """阈值小到连省略标记都放不下时必须放弃，而且不能假装做了事。

    这条是写出来才发现的：原来 purify 在"净化后反而更长"时静默返回原文，
    调用方也静默跳过，于是阈值附近完全不生效却没有任何痕迹。
    """
    text = "a" * 400
    budget = with_overrides(B, tau_len=150)            # avail = 150 - 240 < 0
    out, purged = purify(text, budget)
    assert out == text and purged == 0


def test_tool_signature_is_stable_and_discriminating():
    assert tool_signature("read_file", {"file_path": "a.py"}) == tool_signature("read_file", {"file_path": "a.py"})
    assert tool_signature("read_file", {"file_path": "a.py"}) != tool_signature("read_file", {"file_path": "b.py"})
    assert tool_signature("read_file", {"file_path": "a.py"}) != tool_signature("grep_search", {"file_path": "a.py"})


def test_supervisor_purifies_and_records():
    sup = TaskSupervisor(task_id="sq_001", budget=B)
    text = obs(300)
    out = sup.observe("read_file", {"file_path": "agents/agent.py"}, text)
    assert len(out) < len(text)
    assert sup.purged_total > 0
    assert SIG_EXCESSIVE in [i.signal for i in sup.intercepts]
    assert any(i.action == ACT_PURIFY for i in sup.intercepts)


def test_supervisor_disabled_records_approve_for_ab_comparison():
    """净化关掉时必须留下痕迹，否则对照组看不出差异从哪来。"""
    budget = with_overrides(B, enable_purification=False)
    sup = TaskSupervisor(task_id="sq_001", budget=budget)
    text = obs(300)
    out = sup.observe("read_file", {"file_path": "x.py"}, text)
    assert out == text
    assert sup.purged_total == 0
    assert any(i.signal == SIG_EXCESSIVE and i.action == ACT_APPROVE for i in sup.intercepts)


def test_supervisor_records_when_purification_cannot_shrink():
    budget = with_overrides(B, tau_len=150)
    sup = TaskSupervisor(task_id="sq_001", budget=budget)
    text = "a" * 400
    out = sup.observe("read_file", {"file_path": "x.py"}, text)
    assert out == text
    assert sup.purged_total == 0
    row = [i for i in sup.intercepts if i.signal == SIG_EXCESSIVE]
    assert row and row[0].action == ACT_APPROVE
    assert "未能缩小" in row[0].note


def test_loop_detection_fires_on_repeat_signature():
    sup = TaskSupervisor(task_id="sq_001", budget=B)
    sup.observe("read_file", {"file_path": "a.py"}, "ok")
    sup.observe("read_file", {"file_path": "a.py"}, "ok")
    assert any(i.signal == SIG_LOOP for i in sup.intercepts)


def test_loop_detection_ignores_distinct_calls():
    sup = TaskSupervisor(task_id="sq_001", budget=B)
    sup.observe("read_file", {"file_path": "a.py"}, "ok")
    sup.observe("read_file", {"file_path": "b.py"}, "ok")
    assert not any(i.signal == SIG_LOOP for i in sup.intercepts)


def test_step_check_fires_every_tau_step():
    sup = TaskSupervisor(task_id="sq_001", budget=B)
    for i in range(B.tau_step):
        sup.observe("grep_search", {"pattern": f"p{i}"}, "ok")
    checks = [i for i in sup.intercepts if i.signal == SIG_STEP_CHECK]
    assert len(checks) == 1
    assert checks[0].step == B.tau_step


def test_intercept_serialisation_has_purged_chars():
    sup = TaskSupervisor(task_id="sq_001", budget=B)
    sup.observe("read_file", {"file_path": "x.py"}, obs(300))
    rows = sup.as_dicts()
    assert rows and all("purged_chars" in r for r in rows)
    assert any(r["purged_chars"] > 0 for r in rows)
