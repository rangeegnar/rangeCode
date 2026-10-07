"""接入边界的结构测试 —— 零模型调用。

这里锁的不是功能，是**依赖形状**。三条一旦被破坏，代价是每次构造 Agent 都
多加载一整条编排依赖链，或者 plan 模式能真的跑起编排：

1. `definition.py` 不许出现 langgraph / langchain 之类重依赖的 import；
2. `__init__.py` 不许 re-export 子模块（`import agents.multiagents.definition`
   会先执行它，一旦它 import interface，轻量入口就白拆了）；
3. 工具 schema 的形状要能被 `get_active_tool_definitions` 原样接受。
"""
from __future__ import annotations

import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]

HEAVY = ("langgraph", "langchain", "langchain_core", "langchain_openai",
         "openai", "anthropic", "dotenv", "pydantic", "rich", "tqdm")


def _imported_modules(path: Path) -> list[str]:
    """用 AST 取真正的 import 目标。

    不能用"按行 startswith('from ')"的粗糙办法：docstring 里的示例代码
    会被误判成真 import（这条是实测踩到的，见 test_package_init_*）。
    返回的字符串里，相对导入带前导点（`from .x import y` → `.x`）。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            out.append("." * node.level + (node.module or ""))
    return out


def _relative_imports(path: Path) -> list[str]:
    return [m for m in _imported_modules(path) if m.startswith(".")]


def test_definition_has_no_heavy_imports():
    """工具定义必须能单独 import，不能顺带拉起 langgraph。"""
    modules = _imported_modules(PKG / "definition.py")
    bad = [m for m in modules if m.split(".")[0] in HEAVY]
    assert bad == [], f"definition.py 引入了重依赖：{bad}"


def test_package_init_does_not_reexport_submodules():
    """包 __init__ 一旦 re-export，轻量入口就失效了。

    只允许 `from __future__ import annotations` 这一条。
    """
    modules = _imported_modules(PKG / "__init__.py")
    assert modules == ["__future__"], f"agents/multiagents/__init__.py 不该 import 别的：{modules}"


def test_only_definition_is_importable_cheaply():
    """definition.py 只依赖 typing，且不引同包其它模块。"""
    modules = _imported_modules(PKG / "definition.py")
    assert sorted(modules) == ["__future__", "typing"], modules
    assert _relative_imports(PKG / "definition.py") == []


def test_tool_def_shape_is_accepted_by_tool_loader():
    """工具的键必须只在 BearCode 允许的集合里，否则 _to_openai_tools 会出错。"""
    from agents.multiagents.definition import MULTIAGENT_TOOL_DEF
    from agents.tools import get_active_tool_definitions

    allowed = {"name", "description", "input_schema", "deferred"}
    assert set(MULTIAGENT_TOOL_DEF) <= allowed, set(MULTIAGENT_TOOL_DEF) - allowed

    active = get_active_tool_definitions([MULTIAGENT_TOOL_DEF])
    assert len(active) == 1
    assert "deferred" not in active[0], "deferred 键应被 loader 剥掉"


def test_tool_def_is_not_deferred():
    """不给 deferred 才会一直可见；给了就要求 tools_search 先激活。"""
    from agents.multiagents.definition import MULTIAGENT_TOOL_DEF
    assert not MULTIAGENT_TOOL_DEF.get("deferred")


def test_tool_def_requires_only_task():
    from agents.multiagents.definition import MULTIAGENT_TOOL_DEF
    schema = MULTIAGENT_TOOL_DEF["input_schema"]
    assert schema["required"] == ["task"]
    assert set(schema["properties"]) == {"task", "max_iterations"}


def test_tool_is_not_marked_concurrency_safe():
    """编排层内部已扇出到端点并发上限，标进并发安全集合会撞 429。"""
    from agents.tools import CONCURRENCY_SAFE_TOOLS
    assert "multiagent" not in CONCURRENCY_SAFE_TOOLS


def test_records_dir_is_distinct_from_subagent_definitions():
    """运行记录目录与自定义子代理定义目录必须是两个名字。"""
    from agents.multiagents import records
    from agents.subagent import _load_agents_from_dir  # noqa: F401 只确认可导入

    assert records.RECORDS_DIRNAME == "multiagents"
    assert records.RECORDS_DIRNAME != "subagent"
