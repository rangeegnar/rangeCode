"""接入 `agents/` 之后的边界测试 —— 全是结构与权限断言，零模型调用。

这里锁住的是接入约定本身，不是编排功能：
  - plan 模式必须拦住 multiagent（工具不在 EDIT_TOOLS/READ_TOOLS 里，
    不显式拦就会落到末尾的 allow）；
  - 工具注册的三个条件（非子代理 / 非 plan / OpenAI 协议）；
  - `agents/agent.py` 只许 import 轻量的 definition，不许 import interface；
  - `_chat_openai` 里 `oai_batches` 的构建必须在 `for tc in tool_calls` **之外**
    （历史缺陷：写在循环内会导致模型一轮发多个工具调用时早先结果被重复追加，
    报 `Duplicate value for 'tool_call_id'`，并且工具会被重复执行）。
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
AGENT_PY = REPO_ROOT / "agents" / "agent.py"
PROMPT_PY = REPO_ROOT / "agents" / "prompt.py"

OPENAI_KW = {"model": "ecnu-max",
             "api_base": "https://chat.ecnu.edu.cn/open/api/v1",
             "api_key": "dummy"}


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _func(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"没找到函数 {name}")


def _module_imports(tree: ast.Module) -> list[str]:
    out: list[str] = []
    for node in tree.body:                       # 只看模块顶层
        if isinstance(node, ast.Import):
            out.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            out.append("." * node.level + (node.module or ""))
    return out


# ── 权限 ───────────────────────────────────────────────────
def test_plan_mode_denies_multiagent():
    """plan 模式约定只读，而编排会真的派发 worker 并写运行记录。"""
    from agents.tools import check_permission
    result = check_permission("multiagent", {"task": "x"}, "plan")
    assert result["action"] == "deny", result
    assert "plan mode" in result["message"]


def test_default_mode_allows_multiagent():
    from agents.tools import check_permission
    assert check_permission("multiagent", {"task": "x"}, "default")["action"] == "allow"


def test_bypass_mode_allows_multiagent():
    from agents.tools import check_permission
    assert check_permission("multiagent", {"task": "x"}, "bypassPermissions")["action"] == "allow"


def test_plan_mode_deny_is_explicit_not_fallthrough():
    """拦它必须是一条显式分支，而不是靠末尾兜底的 allow。

    做法：把 multiagent 从 plan 分支里摘掉后行为会变。这里只能断言源码里有这条分支，
    因为它正是在防"落到末尾 allow"。
    """
    src = (REPO_ROOT / "agents" / "tools.py").read_text(encoding="utf-8")
    assert 'if tool_name == "multiagent":' in src
    # 且这条分支出现在 plan 模式块内、run_shell 那条之后
    plan_idx = src.index('if mode == "plan":')
    shell_idx = src.index('if tool_name == "run_shell":', plan_idx)
    ma_idx = src.index('if tool_name == "multiagent":', plan_idx)
    assert plan_idx < shell_idx < ma_idx


# ── 工具注册条件 ───────────────────────────────────────────
def test_tool_registration_is_guarded_by_three_conditions():
    tree = _parse(AGENT_PY)
    init = _func(tree, "__init__")

    target = "self.tools = self.tools + [MULTIAGENT_TOOL_DEF]"
    guard_tests: list[str] = []
    for node in ast.walk(init):
        if not isinstance(node, ast.If):
            continue
        body_src = "\n".join(ast.get_source_segment(
            AGENT_PY.read_text(encoding="utf-8"), stmt) or "" for stmt in node.body)
        if target in body_src:
            guard_tests.append(ast.get_source_segment(
                AGENT_PY.read_text(encoding="utf-8"), node.test) or "")

    assert guard_tests, "没找到 multiagent 的注册语句"
    guard = " ".join(guard_tests)
    for needed in ("is_sub_agent", "plan", "use_openai"):
        assert needed in guard, f"注册条件缺少 {needed}：{guard}"


def test_agent_py_imports_definition_not_interface():
    """构造 Agent 时必须走轻量入口，否则每次都拉起 langgraph。"""
    imports = _module_imports(_parse(AGENT_PY))
    assert "agents.multiagents.definition" in imports, imports
    assert "agents.multiagents.interface" not in imports, imports
    assert "agents.multiagents" not in imports, "不该 import 包本身（会拉起 __init__ 之外的判断）"


def test_interface_is_imported_lazily_inside_the_executor():
    src = AGENT_PY.read_text(encoding="utf-8")
    body = src[src.index("async def _execute_multiagent_tool"):]
    assert "from agents.multiagents.interface import execute_multiagent_with_usage" in body


# ── 历史缺陷回归：多工具调用 ────────────────────────────────
def _assigns_to(node: ast.AST, prefix: str) -> list[ast.AST]:
    """找出对 `prefix*` 的赋值语句。

    必须同时认 `Assign` 和 `AnnAssign`：源码里写的是
    `oai_batches: list[dict] = []`，那是 AnnAssign，只查 Assign 会一个都找不到
    （这条踩过一次，检查本身失效会伪装成"代码不见了"）。
    """
    out: list[ast.AST] = []
    for n in ast.walk(node):
        if isinstance(n, ast.Assign):
            targets: list[ast.expr] = list(n.targets)
        elif isinstance(n, ast.AnnAssign):
            targets = [n.target]
        else:
            continue
        if any(isinstance(t, ast.Name) and t.id.startswith(prefix) for t in targets):
            out.append(n)
    return out


def test_oai_batches_is_built_outside_the_tool_call_loop():
    """`oai_batches` 必须在 `for tc in tool_calls:` 退出之后才构建。

    写在循环内时：每处理一个 tc 就按"已累积的 oai_checked"重建整批，
    于是早先的工具被重复执行、tool 消息被重复追加，接口直接 400。
    """
    tree = _parse(AGENT_PY)
    chat = _func(tree, "_chat_openai")

    loops = [n for n in ast.walk(chat)
             if isinstance(n, ast.For) and isinstance(n.iter, ast.Name)
             and n.iter.id == "tool_calls"]
    assert len(loops) == 1, f"预期恰好一个 tool_calls 循环，找到 {len(loops)} 个"
    loop = loops[0]

    nested = _assigns_to(loop, "oai_batches")
    assert nested == [], (
        f"oai_batches 仍然在 for tc in tool_calls 里面（第 {nested[0].lineno} 行）")

    # 而且要确实存在，别是被我整段删掉了
    every = _assigns_to(chat, "oai_batches")
    assert every, "oai_batches 不见了 —— 反缩进时把代码删了？"


def test_tool_results_are_appended_once_per_call():
    """`oai_checked` 的构建同样必须在循环外，否则工具会被重复执行。"""
    tree = _parse(AGENT_PY)
    chat = _func(tree, "_chat_openai")
    loops = [n for n in ast.walk(chat)
             if isinstance(n, ast.For) and isinstance(n.iter, ast.Name)
             and n.iter.id == "tool_calls"]
    assert _assigns_to(loops[0], "oai_checked") == [], "oai_checked 还在循环内"


def test_multiagent_is_reachable_from_the_tool_dispatch():
    """`_execute_tool_call` 里必须有 multiagent 分支，否则工具永远跑不起来。"""
    src = AGENT_PY.read_text(encoding="utf-8")
    dispatch = src[src.index("async def _execute_tool_call"):]
    dispatch = dispatch[:dispatch.index("async def ", 10)]
    assert 'if name == "multiagent":' in dispatch
    assert "_execute_multiagent_tool(inp)" in dispatch


# ── 提示词 ─────────────────────────────────────────────────
def test_prompt_tells_the_model_when_to_use_multiagent():
    src = PROMPT_PY.read_text(encoding="utf-8")
    assert "`multiagent`" in src
    # 必须写明"先读完成度报告"和失败时的处置，否则模型会照抄结论
    assert "completeness report" in src
    assert "Multiagent run failed" in src


# ── 真实构造 Agent，验证注册矩阵（不联网）──────────────────
@pytest.mark.parametrize("case,kwargs,expected", [
    ("默认（OpenAI 协议，非子代理）", {}, True),
    ("plan 模式", {"permission_mode": "plan"}, False),
    ("子代理", {"is_sub_agent": True}, False),
    ("Anthropic 协议（编排层只支持 OpenAI 协议）",
     {"model": "claude-x", "anthropic_base_url": "https://api.anthropic.com"}, False),
])
def test_tool_registration_matrix(case, kwargs, expected):
    from agents.agent import Agent

    params = {**OPENAI_KW, **kwargs}
    if "anthropic_base_url" in kwargs:
        params = {"model": kwargs["model"], "anthropic_base_url": kwargs["anthropic_base_url"],
                  "api_key": "dummy"}
    agent = Agent(**params)
    names = [t["name"] for t in agent.tools]
    assert ("multiagent" in names) is expected, f"{case}：{names}"


def test_importing_agents_agent_does_not_pull_langgraph():
    """轻量入口必须真的有效：`import agents.agent` 不能拉起编排依赖链。

    这条必须在**子进程**里跑 —— 同进程的 sys.modules 已经被别的测试污染过了。
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p and Path(p).is_dir())
    code = (
        "import sys, agents.agent\n"
        "heavy = sorted(m for m in sys.modules if m.split('.')[0] in "
        "('langgraph', 'langchain', 'langchain_core', 'langchain_openai'))\n"
        "print('HEAVY:', heavy)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                          env=env, capture_output=True, text=True, timeout=120)
    assert "HEAVY: []" in proc.stdout, proc.stdout + proc.stderr
