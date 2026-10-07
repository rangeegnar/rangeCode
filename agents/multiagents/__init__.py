"""multiagents —— 多领域多智能体编排层（LangGraph 实现）。

编排拓扑取自 VMAO（arXiv:2603.11445）：Plan → Execute → Verify → Replan → Synthesize
防线按 MAST（arXiv:2503.13657）的 14 条失败模式布置
监督层取自 SupervisorAgent（arXiv:2510.26585），挂在 worker 的工具通道上

双模型：
    执行（planner / worker / synthesizer） → MODEL（或宿主 Agent 注入的模型）
    验证（verifier / replanner）           → MODEL4TEST，必须与执行模型不同

**本文件刻意不 import 任何子模块。** 因为 `import agents.multiagents.definition`
会先执行本文件，如果这里 `from .interface import ...`，那么
`agents/agent.py::__init__` 注册工具时就会把整条编排依赖链
（langgraph → langchain → langchain_openai）一起拉进来，每次构造 Agent 都付这个代价。

用到什么就显式 import 对应的子模块：

    from agents.multiagents.definition import MULTIAGENT_TOOL_DEF   # 轻，只有 schema
    from agents.multiagents.interface import execute_multiagent    # 重，会拉起 langgraph
    from agents.multiagents.records import session_dir, load_index # 落盘
    from agents.multiagents.runner import run_orchestration        # 只跑图，不落盘
"""
from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
