"""worker 用的工具集。

两类：
1. 转调项目自身的工具实现（agents/tools.py::execute_tool）——
   read_file / list_files / grep_search。这样 worker 拿到的读文件语义
   （行号格式、路径解析、结果截断）与主 agent-loop 完全一致，不会有两套口径。
2. 编排层自带 —— read_lines（按行区间读，项目没有）、repo_metrics（量化统计）。

刻意不提供：write_file / edit_file / run_shell / skill_* / agent。
写权限和 shell 逃逸面一律不给自动生成的子任务，也不能递归开子代理。
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

from langchain_core.tools import StructuredTool, tool

from .config import PROJECT_ROOT

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", "deps",
             ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def _abs(path: str) -> str:
    """相对路径解析到项目根。

    编排器自己的 cwd 是实验目录，而模型嘴里的 "agents/subagent.py" 是相对项目根的。
    不做这层解析，worker 会去实验目录找 agents/，然后如实报告"文件不存在"——
    这是上一轮实测踩到的坑。
    """
    p = Path(path or ".")
    return str(p) if p.is_absolute() else str((PROJECT_ROOT / p).resolve())


async def _delegate(name: str, inp: dict) -> str:
    from agents.tools import execute_tool

    return await execute_tool(name, inp, {})


@tool
async def read_file(file_path: str) -> str:
    """读取文本文件全文，返回带行号内容。大文件请优先用 read_lines。

    Args:
        file_path: 相对项目根的路径，例如 "agents/subagent.py"。
    """
    return await _delegate("read_file", {"file_path": _abs(file_path)})


@tool
async def list_files(pattern: str) -> str:
    """按 glob 模式列出文件，起点固定为项目根。

    Args:
        pattern: 相对项目根的 glob，例如 "agents/**/*.py" 或 "wiki/*.md"。
    """
    return await _delegate("list_files", {"pattern": pattern, "path": _abs(".")})


@tool
async def grep_search(pattern: str, path: str = ".") -> str:
    """在文件内容里做正则检索，返回命中行。

    Args:
        pattern: 正则表达式，例如 "READ_ONLY_TOOLS|CONCURRENCY_SAFE_TOOLS"。
        path: 相对项目根的检索起点，默认项目根。
    """
    return await _delegate("grep_search", {"pattern": pattern, "path": _abs(path)})


@tool
def read_lines(file_path: str, start: int = 1, end: int = 200) -> str:
    """按行区间读取文件，带行号。大文件用这个，不要用 read_file 全文读。

    Args:
        file_path: 相对项目根的路径，例如 "agents/agent.py"。
        start: 起始行号，从 1 开始。
        end: 结束行号（含边界）；单次最多返回 400 行。
    """
    path = Path(_abs(file_path))
    if not path.is_file():
        return json.dumps({"error": f"文件不存在：{path}"}, ensure_ascii=False)

    start = max(1, int(start))
    end = max(start, int(end))
    if end - start + 1 > 400:
        end = start + 399

    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            window = list(itertools.islice(fh, start - 1, end))
            fh.seek(0)
            total = sum(1 for _ in fh)
    except OSError as exc:
        return json.dumps({"error": f"读取失败：{exc}"}, ensure_ascii=False)

    body = "\n".join(f"{i:5d} | {line.rstrip()}" for i, line in enumerate(window, start=start))
    return f"{file_path} 第 {start}-{start + len(window) - 1} 行 / 共 {total} 行\n{body}"


@tool
def repo_metrics(root: str = ".", extensions: str = "", top_files: int = 10) -> str:
    """统计目录下的文件数量、总行数与体积分布，JSON 返回。

    注意：只给目录级总行数与逐文件**字节数**，不提供逐文件行数。

    Args:
        root: 相对项目根的统计起点，默认项目根。例如 "agents" 只统计该目录。
        extensions: 逗号分隔的扩展名白名单，例如 ".py,.md"；留空统计全部。
        top_files: 返回体积最大的前 N 个文件。
    """
    base = Path(_abs(root))
    if not base.is_dir():
        return json.dumps({"error": f"目录不存在：{base}"}, ensure_ascii=False)

    wanted: set[str] | None = None
    if extensions.strip():
        wanted = {
            (e.strip() if e.strip().startswith(".") else "." + e.strip()).lower()
            for e in extensions.split(",") if e.strip()
        }

    total_files = total_bytes = total_lines = binary_files = 0
    by_ext: dict[str, dict[str, int]] = {}
    largest: list[tuple[int, str]] = []

    for path in base.rglob("*"):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.parts):
            continue
        ext = path.suffix.lower() or "<no-ext>"
        if wanted is not None and ext not in wanted:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        try:
            text = path.read_text(encoding="utf-8")
            lines = text.count("\n") + (1 if text else 0)
        except (UnicodeDecodeError, OSError):
            binary_files += 1
            lines = 0
        total_files += 1
        total_bytes += size
        total_lines += lines
        bucket = by_ext.setdefault(ext, {"files": 0, "lines": 0, "bytes": 0})
        bucket["files"] += 1
        bucket["lines"] += lines
        bucket["bytes"] += size
        largest.append((size, str(path.relative_to(base))))

    largest.sort(reverse=True)
    return json.dumps(
        {
            "root": str(base),
            "extension_filter": sorted(wanted) if wanted else None,
            "skipped_dirs": sorted(SKIP_DIRS),
            "total_files": total_files,
            "total_lines": total_lines,
            "total_bytes": total_bytes,
            "binary_files": binary_files,
            "note": "只有目录级 total_lines 与逐文件字节数，没有逐文件行数",
            "by_extension": dict(sorted(by_ext.items(), key=lambda kv: kv[1]["lines"], reverse=True)),
            "largest_files": [{"path": p, "bytes": s} for s, p in largest[: max(0, top_files)]],
        },
        ensure_ascii=False,
        indent=2,
    )


TOOL_REGISTRY = {
    "read_file": read_file,
    "read_lines": read_lines,
    "list_files": list_files,
    "grep_search": grep_search,
    "repo_metrics": repo_metrics,
}


def wrap_with_supervisor(t, task_supervisor):
    """把工具包一层，让每次调用的返回先过监督层。

    这是监督层挂在「工具通道」上的实际做法，对应 SupervisorAgent 论文的 Agent-Tool 通道。
    """

    async def _run(**kwargs):
        observation = await t.ainvoke(kwargs)
        if not isinstance(observation, str):
            observation = str(observation)
        return task_supervisor.observe(t.name, kwargs, observation)

    return StructuredTool.from_function(
        coroutine=_run,
        name=t.name,
        description=t.description,
        args_schema=t.args_schema,
    )
