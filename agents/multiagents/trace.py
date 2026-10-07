"""节点级运行日志 —— 看关键信息，不打印 state 的 JSON。

设计取舍：

1. **每个节点一行主记录，缩进行放细节。** 并行 worker 交错输出时，
   行首的单调时钟能看出谁先谁后；一整行一次 write，不会互相插字。
2. **默认关闭，由调用方打开。** 这样：
   - pytest 跑离线测试时不会刷屏
   - 脚本直接调 `execute_multiagent` 时不会乱印
   打开方式：`set_enabled(True)`，或走宿主时用 `progress=True`（见 interface）。
3. **不引入日志框架、不写文件。** 完整轨迹在 `.range/multiagents/<session>/*.json`，
   这里只负责"人在看终端时能跟上"。
4. **不打印 state。** 每个节点自己知道哪些字段值得看，直接调 `node()/detail()` 传文本。
5. **渲染可插拔。** 默认写纯文本到 stdout；在宿主（REPL）里调 `use_host_ui()`
   改成走 `agents/ui.py` 的配色渲染。编排层不硬依赖 `agents.ui` ——
   那句 import 在函数里，只有真用的时候才发生。

输出形状（宿主渲染下）：

    ◆ multiagent plan       拆出 4 个子问题  tok=1308  3.4s  ecnu-max
                               sq_001 [code] pri=9 deps=-  agents/subagent.py 里内置了...
    ▸ multiagent worker     sq_001 [code] attempt=1 ok 23.4s tok=38412 净化-5120字
    ✓ multiagent verify     第 1 轮  完成度 0.85  (3/4 complete)  矛盾 2  tok=8171  3.1s  ecnu-plus
"""
from __future__ import annotations

import sys
import time
from typing import Callable

_enabled = False
_t0 = time.monotonic()

NODE_WIDTH = 10

# 可插拔渲染：宿主给一个 (node_fn, detail_fn)，不给就用下面的纯文本实现
_node_printer: Callable[[str, str], None] | None = None
_detail_printer: Callable[[str], None] | None = None


def set_enabled(flag: bool) -> None:
    global _enabled
    _enabled = bool(flag)


def enabled() -> bool:
    return _enabled


def set_printer(node_fn: Callable[[str, str], None] | None,
                detail_fn: Callable[[str], None] | None = None) -> None:
    """换掉渲染方式。传 None 恢复纯文本输出。"""
    global _node_printer, _detail_printer
    _node_printer = node_fn
    _detail_printer = detail_fn if detail_fn is not None else node_fn


def use_host_ui() -> None:
    """把输出接到宿主 `agents/ui.py` 的渲染上。

    编排层不硬依赖 ui.py：这句 import 放在函数里，只有真打开宿主渲染时才发生，
    所以单元测试和脚本直连都不需要 rich / 终端环境。
    """
    from agents.ui import print_multiagent_detail, print_multiagent_node

    set_printer(print_multiagent_node, print_multiagent_detail)


def reset() -> None:
    """把时钟零点挪到现在。每次运行前调一次，行首的秒数才是本次运行的耗时。"""
    global _t0
    _t0 = time.monotonic()


def _stamp() -> str:
    return f"{time.monotonic() - _t0:7.1f}s"


def _emit(text: str) -> None:
    if not _enabled:
        return
    # 一整行（含换行）一次 write，避免并行 worker 之间互相插字
    sys.stdout.write(f"[{_stamp()}] {text}\n")
    sys.stdout.flush()


def node(name: str, text: str) -> None:
    """一个节点的一行主记录。"""
    if not _enabled:
        return
    if _node_printer is not None:
        _node_printer(name, text)
        return
    _emit(f"{name:<{NODE_WIDTH}} {text}")


def detail(text: str) -> None:
    """节点的细节行，缩进到主记录文本下方。"""
    if not _enabled:
        return
    if _detail_printer is not None:
        _detail_printer(text)
        return
    _emit(f"{'':<{NODE_WIDTH}}   {text}")


def ids(items) -> str:
    """把 id 列表压成一行，空的时候给 "-"。"""
    seq = [str(i) for i in (items or [])]
    return ",".join(seq) if seq else "-"


def truncate(text: str, width: int) -> str:
    """单行展示用。换行压成空格，超长截断并标注。"""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"
