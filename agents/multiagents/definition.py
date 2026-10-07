"""工具定义 —— 只有 schema，**零重依赖**。

这个模块必须能单独被 import 而不拉起 langgraph / langchain。
原因：`agents/agent.py::__init__` 需要把工具注册进 `self.tools`，
如果那里 import 的是 `interface.py`，那么每构造一个 Agent 就会把
整条编排依赖链（langgraph → langchain → langchain_openai）加载进来。

所以拆成两个模块：
    definition.py  只有 dict 字面量，无 import
    interface.py   真正执行编排，延迟到工具被调用时才 import

注意：包名是 `multiagents`（复数，因为里面是一个团队），工具名是
`multiagent`（单数，因为对主 agent 来说它是一个工具）。这是刻意的。
"""
from __future__ import annotations

from typing import Any

MULTIAGENT_TOOL_DEF: dict[str, Any] = {
    "name": "multiagent",
    "description": (
        "Launch a team of domain-specialised sub-agents to work a task in parallel, "
        "then independently verify (with a different model) and synthesise their findings. "
        "Use it when a task spans multiple domains (code reading, doc lookup, repo metrics, "
        "writing) or needs several independent investigations whose results must be "
        "cross-checked. Returns a completeness report first, then the synthesised answer — "
        "the report lists gaps and contradictions, so read it before trusting the answer. "
        "For a single narrow question, use the agent tool instead — this one costs more "
        "(typically 10x a single sub-agent) and takes 1-3 minutes."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": (
                    "The full task to hand to the team. State the goal and what a good "
                    "answer must contain; the planner uses this to split the work."
                ),
            },
            "max_iterations": {
                "type": "integer",
                "description": "Override the verify/replan iteration cap. Default 2. Higher costs more.",
            },
        },
        "required": ["task"],
    },
}

__all__ = ["MULTIAGENT_TOOL_DEF"]
