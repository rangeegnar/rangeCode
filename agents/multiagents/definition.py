"""
工具定义 —— 只有 schema。
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
