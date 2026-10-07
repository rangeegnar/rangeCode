"""项目级配置常量的唯一来源。

**本模块只依赖标准库，绝不 import 任何 `agents.*` 模块。** 这一条是硬约束：
`agents/agent.py`、`agents/tools.py`、`agents/main.py`、`agents/multiagents/*`
都会 import 它，任何反向依赖都会立刻形成循环导入。

内容分四块：

1. `clean_env()` —— 环境变量清洗。全项目解析 .env 的唯一出口。
2. 模型容量 —— 上下文窗口 / 单次最大输出。按**角色**（执行槽位 `MODEL`、
   验证槽位 `MODEL4TEST`）从 .env 读，读不到用本模块默认值，模型不认识则用通用兜底。
3. 上下文压缩与工具结果截断的阈值 —— 主 agent-loop（`agents/agent.py`）用。
4. 重试退避与 side query 的 token 上限。

本模块只读 `os.environ`，**不自己 load_dotenv** —— 加载时机由入口决定
（`agents/main.py` 启动时已加载）。这样它在单元测试里也能被干净地打桩。
"""
from __future__ import annotations

import os

# ══════════════════════════════════════════════════════════════
# 1. 环境变量清洗
# ══════════════════════════════════════════════════════════════

def clean_env(value: str | None) -> str | None:
    """去掉首尾空白与行尾注释，空串归一成 None。

    必须剥行尾注释：dotenv 会把 `KEY=value # note` 的**整行**（含注释）放进
    `os.environ`。项目历史上出过一次污染事故（注释被当成模型名发出去），
    所以全项目统一走这一个函数。

    `.env` 里那几行的真实写法：

        MODEL_MAX_CONTEXT=524288        # ecnu-max 512K = 512*1024

    不剥注释，`int()` 会拿到 `"524288        # ecnu-max 512K = 512*1024"` 并抛
    ValueError，于是配置**静默失效**退回默认值。
    """
    if value is None:
        return None
    return value.split("#")[0].strip() or None


# ══════════════════════════════════════════════════════════════
# 2. 模型容量：上下文窗口 + 单次最大输出
# ══════════════════════════════════════════════════════════════

# ── 按角色的默认值（.env 没配对应键时用这些）──────────────────
# 执行模型 512K 上下文 / 384K 输出；验证模型 256K / 128K。
EXEC_DEFAULT_CONTEXT = 524_288        # 512 * 1024
EXEC_DEFAULT_MAX_OUTPUT = 393_216     # 384 * 1024
VERIFY_DEFAULT_CONTEXT = 262_144      # 256 * 1024
VERIFY_DEFAULT_MAX_OUTPUT = 131_072   # 128 * 1024

# ── 通用兜底 ───────────────────────────────────────────────
# 既不属于执行槽位、也不属于验证槽位的模型走这一对。典型场景：
#   - 单元测试里随便起的模型名
#   - `Agent(model=...)` 或 CLI `--model` 临时换了第三个模型
# 这里**刻意不内置任何厂商模型表**：上下文窗口是服务端属性，写死在代码里
# 只会随时间过期，而过期值会让四层压缩在错误的时机触发（比没有值更糟）。
# 真要用第三个模型，请在 .env 里显式配一对上限。
UNKNOWN_CONTEXT = 200_000
UNKNOWN_MAX_OUTPUT = 16_384

_ENV_EXEC_MODEL = "MODEL"
_ENV_VERIFY_MODEL = "MODEL4TEST"
_ENV_EXEC_CONTEXT = "MODEL_MAX_CONTEXT"
_ENV_EXEC_OUTPUT = "MODEL_MAX_OUTPUT"
_ENV_VERIFY_CONTEXT = "MODEL4TEST_MAX_CONTEXT"
_ENV_VERIFY_OUTPUT = "MODEL4TEST_MAX_OUTPUT"


def _as_positive_int(value: str | None) -> int | None:
    """转成正整数；非法值（None / 非数字 / <=0）一律返回 None 交给调用方兜底。"""
    if value is None:
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _role_of(model: str) -> str | None:
    """`model` 属于哪个槽位：'exec' / 'verify' / None（不认识）。

    两个槽位同名时按 exec 处理 —— 那说明根本没配验证模型。
    """
    exec_model = clean_env(os.environ.get(_ENV_EXEC_MODEL))
    if exec_model and model == exec_model:
        return "exec"
    verify_model = clean_env(os.environ.get(_ENV_VERIFY_MODEL))
    if verify_model and model == verify_model:
        return "verify"
    return None


def context_window(model: str) -> int:
    """该模型的上下文窗口（token 数）。"""
    role = _role_of(model)
    if role == "exec":
        configured = _as_positive_int(clean_env(os.environ.get(_ENV_EXEC_CONTEXT)))
        return configured if configured is not None else EXEC_DEFAULT_CONTEXT
    if role == "verify":
        configured = _as_positive_int(clean_env(os.environ.get(_ENV_VERIFY_CONTEXT)))
        return configured if configured is not None else VERIFY_DEFAULT_CONTEXT
    return UNKNOWN_CONTEXT


def max_output_tokens(model: str) -> int:
    """该模型单次返回的最大 token 数。"""
    role = _role_of(model)
    if role == "exec":
        configured = _as_positive_int(clean_env(os.environ.get(_ENV_EXEC_OUTPUT)))
        return configured if configured is not None else EXEC_DEFAULT_MAX_OUTPUT
    if role == "verify":
        configured = _as_positive_int(clean_env(os.environ.get(_ENV_VERIFY_OUTPUT)))
        return configured if configured is not None else VERIFY_DEFAULT_MAX_OUTPUT
    return UNKNOWN_MAX_OUTPUT


def describe(model: str) -> dict[str, int | str]:
    """一行说明，给 dry-run / 启动日志用。"""
    return {"model": model, "role": _role_of(model) or "unknown",
            "context_window": context_window(model),
            "max_output_tokens": max_output_tokens(model)}


# ══════════════════════════════════════════════════════════════
# 3. 上下文压缩与工具结果截断
# ══════════════════════════════════════════════════════════════
# 一个工具结果进上下文要依次过这几道闸，改动时注意它们是叠加的：
#   MAX_RESULT_CHARS → PERSIST_RESULT_THRESHOLD_BYTES
#   → BUDGET_THRESHOLD → SNIP_THRESHOLD → MICROCOMPACT_IDLE_S → AUTO_COMPACT_THRESHOLD

# 从窗口里给模型回复留出的余量。注意**不要**改成 max_output_tokens：
# 输出上限是 393216，拿它当预留会让 effective_window 掉到 11 万，
# 反而比收紧前更早触发压缩。
RESPONSE_RESERVE_TOKENS = 20_000

# 单个工具结果进入上下文前的硬上限（`agents/tools.py::_truncate_result`）。
# 覆盖几乎所有工具：read_file / grep_search / list_files / run_shell / write_file 等。
MAX_RESULT_CHARS = 50_000
# 截断时中间那行 `[... truncated N chars ...]` 预留的字符数，
# 保证截断后的总长仍然 <= MAX_RESULT_CHARS。
TRUNCATE_MARKER_RESERVE_CHARS = 60

# 大结果落盘（`agents/agent.py::_persist_large_result`）。
# 注意这道比 MAX_RESULT_CHARS 更紧：50000 字符截完仍可能超过 30KB，
# 于是又变成"只留前 200 行预览"。两处要一起看。
PERSIST_RESULT_THRESHOLD_BYTES = 30 * 1024      # 30 KB
PERSIST_PREVIEW_LINES = 200                     # 落盘后上下文里保留的行数

# 第 3 道：历史工具结果重新截断的触发线（占 effective_window 的比例）。
BUDGET_THRESHOLD = 0.50
# 第 4 道：把历史结果整条换成 SNIP_PLACEHOLDER 的触发线。
SNIP_THRESHOLD = 0.60
# 第 6 道：交给 TierMem 归档的触发线。
AUTO_COMPACT_THRESHOLD = 0.70

SNIP_PLACEHOLDER = "[Content snipped - re-read if needed]"
# 第 5 道（微压缩）留下的占位文本。
OLD_RESULT_CLEARED = "[Old result cleared]"
# 只有这几个工具的历史结果会被整条删掉 —— 都是"可以重新读一次"的只读工具。
SNIPPABLE_TOOLS = {"read_file", "grep_search", "list_files", "run_shell"}
# 第 5 道：距上次 API 调用超过这么久（秒）就清一次老结果。
MICROCOMPACT_IDLE_S = 5 * 60
# 第 4、5 道都要保留最近 N 条不动。
KEEP_RECENT_RESULTS = 3


# ══════════════════════════════════════════════════════════════
# 4. 重试退避 与 side query
# ══════════════════════════════════════════════════════════════

# `agents/agent.py::_with_retry`。延迟 = min(BASE * 2**attempt, CAP) + 抖动。
RETRY_MAX_ATTEMPTS = 3
RETRY_BACKOFF_BASE_MS = 1_000
RETRY_BACKOFF_CAP_MS = 30_000

# `agents/agent.py::_build_side_query` 的默认输出上限，以及三个调用点的语义化取值。
SIDE_QUERY_MAX_TOKENS = 256
SIDE_QUERY_ARCHIVE_MAX_TOKENS = 2_200     # 归档/总结类
SIDE_QUERY_EXTRACT_MAX_TOKENS = 700       # 信息提取类


__all__ = [
    # 环境变量
    "clean_env",
    # 模型容量
    "EXEC_DEFAULT_CONTEXT", "EXEC_DEFAULT_MAX_OUTPUT",
    "VERIFY_DEFAULT_CONTEXT", "VERIFY_DEFAULT_MAX_OUTPUT",
    "UNKNOWN_CONTEXT", "UNKNOWN_MAX_OUTPUT",
    "context_window", "max_output_tokens", "describe",
    # 上下文压缩与结果截断
    "RESPONSE_RESERVE_TOKENS", "MAX_RESULT_CHARS", "TRUNCATE_MARKER_RESERVE_CHARS",
    "PERSIST_RESULT_THRESHOLD_BYTES", "PERSIST_PREVIEW_LINES",
    "BUDGET_THRESHOLD", "SNIP_THRESHOLD", "AUTO_COMPACT_THRESHOLD",
    "SNIP_PLACEHOLDER", "OLD_RESULT_CLEARED", "SNIPPABLE_TOOLS",
    "MICROCOMPACT_IDLE_S", "KEEP_RECENT_RESULTS",
    # 重试与 side query
    "RETRY_MAX_ATTEMPTS", "RETRY_BACKOFF_BASE_MS", "RETRY_BACKOFF_CAP_MS",
    "SIDE_QUERY_MAX_TOKENS", "SIDE_QUERY_ARCHIVE_MAX_TOKENS", "SIDE_QUERY_EXTRACT_MAX_TOKENS",
]
