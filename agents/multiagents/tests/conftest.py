"""pytest 引导：把仓库根加进 sys.path。

测试放在 `agents/multiagents/tests/`，模块是 `agents.multiagents.*`，
所以需要仓库根在 sys.path 上（`agents` 是 PEP420 命名空间包，没有 __init__.py）。

依赖（langgraph / langchain / langchain-openai 等）由项目的 .venv 提供，
因此这个文件**不再往 sys.path 里塞实验目录的 deps/**。
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]   # tests -> multiagents -> agents -> repo
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
