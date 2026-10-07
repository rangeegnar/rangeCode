"""监督层 —— SupervisorAgent（arXiv:2510.26585）的落点。

挂在**工具调用通道**上，不是挂在图上。原因：LangGraph 管的是节点之间的流转，
管不到单次工具调用返回的 5 万字符观测。而 agents/tools.py 的 MAX_RESULT_CHARS = 50000，
读一次 agents/agent.py（99KB）就是约 1.5 万 token，工具循环每一轮都会把它重发一遍。

本模块**全部是纯函数与本地状态，不做任何模型调用**：过滤器是启发式的（照抄论文），
净化是确定性的字符串处理。这样才能逐字复现、零额外成本，也才能被单测覆盖。

阈值照抄论文附录 Table 5：τ_len=3000 字符、τ_loop=5 步、τ_step=8 步。

四级动作按强度：approve < provide_guidance < correct_observation < run_verification。
本项目默认只开 correct_observation 里的「长度净化」。
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .config import Budget

SIG_EXCESSIVE = "excessive_observation"
SIG_LOOP = "loop_detected"
SIG_ERROR = "error_occurred"
SIG_STEP_CHECK = "step_check"

ACT_APPROVE = "approve"
ACT_PURIFY = "correct_observation"
ACT_GUIDANCE = "provide_guidance"
ACT_VERIFY = "run_verification"

_BLANK_RUN_RE = re.compile(r"\n{3,}")

# 给省略标记预留的字符预算。标记本身约 200 字符，留 240 保证净化后总长度不超过 τ_len。
# 不留这个余量会出现"净化后反而更长"从而被放弃的情况（曾在阈值附近静默不生效）。
_MARKER_RESERVE = 240


def tool_signature(name: str, inp: dict[str, Any]) -> str:
    """工具调用的稳定签名，用于循环检测。

    只取工具名与参数形状（键 + 值前 120 字符），不取全文 —— 否则参数里带
    时间戳或随机内容就永远检测不到重复。
    """
    parts = [name]
    for key in sorted(inp or {}):
        parts.append(f"{key}={str(inp[key])[:120]}")
    return "|".join(parts)


def _looks_like_error(observation: str) -> bool:
    head = observation[:400].lower()
    return any(m in head for m in (
        "error:", "error ", "traceback", "no such file", "not found",
        "permission denied", "failed to", "unknown tool"))


def is_excessive(observation: str, budget: Budget) -> bool:
    return len(observation) > budget.tau_len


def purify(observation: str, budget: Budget) -> tuple[str, int]:
    """确定性净化：保头保尾、压空行，**保留行号前缀**。

    返回 (新观测, 省掉的字符数)。省不掉就返回原串与 0。

    刻意不做的事：
    - 不做摘要（那是模型调用，且论文承认过度压缩伤性能）
    - 不删行号（代码阅读依赖行号定位，属于论文说的「噪声即信号」）
    - 不改写内容（只丢弃，不生成）
    """
    original_len = len(observation)
    if original_len <= budget.tau_len:
        return observation, 0

    collapsed = _BLANK_RUN_RE.sub("\n\n", observation)
    if len(collapsed) <= budget.tau_len:
        return collapsed, original_len - len(collapsed)

    # 标记本身的长度要从预算里扣掉，否则净化后可能反而更长
    avail = budget.tau_len - _MARKER_RESERVE
    if avail <= 0:
        # 阈值小到连标记都放不下：明确放弃，由调用方记录
        return observation, 0

    head_budget = int(avail * 0.60)
    tail_budget = avail - head_budget
    head = collapsed[:head_budget]
    tail = collapsed[-tail_budget:]
    dropped = len(collapsed) - head_budget - tail_budget

    marker = (
        f"\n\n[... 监督层净化：省略中间 {dropped} 字符 "
        f"（原文共 {original_len} 字符，超过阈值 {budget.tau_len}）。"
        f"需要省略部分时用 read_lines 指定行区间重新读取。 ...]\n\n"
    )
    result = head + marker + tail
    purged = original_len - len(result)
    if purged <= 0:
        return observation, 0
    return result, purged


@dataclass
class Intercept:
    task_id: str
    step: int
    signal: str
    action: str
    note: str
    before_chars: int
    after_chars: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "step": self.step,
            "signal": self.signal,
            "action": self.action,
            "note": self.note,
            "before_chars": self.before_chars,
            "after_chars": self.after_chars,
            "purged_chars": max(0, self.before_chars - self.after_chars),
        }


@dataclass
class TaskSupervisor:
    """单个子任务的监督状态。每个 worker 一个实例，互不干扰。"""

    task_id: str
    budget: Budget
    step: int = 0
    recent: deque[str] = field(default_factory=lambda: deque(maxlen=8))
    intercepts: list[Intercept] = field(default_factory=list)
    purged_total: int = 0

    def observe(self, tool_name: str, inp: dict[str, Any], observation: str) -> str:
        """看一次工具调用与它的返回。返回可能被替换掉的观测（不替换就原样返回）。"""
        self.step += 1
        sig = tool_signature(tool_name, inp)
        loop_hit = sig in self.recent
        self.recent.append(sig)

        # 1) 观测过长 —— 唯一默认开启的动作
        if is_excessive(observation, self.budget):
            if self.budget.enable_purification:
                new_obs, purged = purify(observation, self.budget)
                if purged > 0:
                    self.purged_total += purged
                    self._record(SIG_EXCESSIVE, ACT_PURIFY,
                                 f"{tool_name} 返回 {len(observation)} 字符，净化后 {len(new_obs)}",
                                 len(observation), len(new_obs))
                    observation = new_obs
                else:
                    # 开了净化但没能省下字符（阈值太小、标记放不下）：必须留记录，
                    # 否则就是静默不生效，对照实验会误判成"净化没用"。
                    self._record(SIG_EXCESSIVE, ACT_APPROVE,
                                 f"{tool_name} 返回 {len(observation)} 字符，超过阈值 "
                                 f"{self.budget.tau_len} 但净化未能缩小（标记开销 >= 可压缩量）",
                                 len(observation), len(observation))
            else:
                self._record(SIG_EXCESSIVE, ACT_APPROVE,
                             f"{tool_name} 返回 {len(observation)} 字符，净化已关闭（对照模式）",
                             len(observation), len(observation))

        # 2) 重复调用循环
        if loop_hit:
            action = ACT_GUIDANCE if self.budget.enable_guidance else ACT_APPROVE
            self._record(SIG_LOOP, action,
                         f"{tool_name} 的调用签名在最近 {len(self.recent)} 步内出现过：{sig[:160]}",
                         len(observation), len(observation))

        # 3) 显式错误
        if _looks_like_error(observation):
            action = ACT_VERIFY if self.budget.enable_verification_action else ACT_APPROVE
            self._record(SIG_ERROR, action,
                         f"{tool_name} 返回疑似错误：{observation[:200]!r}",
                         len(observation), len(observation))

        # 4) 周期性低效检查
        if self.step % self.budget.tau_step == 0:
            self._record(SIG_STEP_CHECK, ACT_APPROVE,
                         f"已执行 {self.step} 步，累计净化 {self.purged_total} 字符",
                         len(observation), len(observation))

        return observation

    def _record(self, signal: str, action: str, note: str, before: int, after: int) -> None:
        self.intercepts.append(Intercept(self.task_id, self.step, signal, action, note, before, after))

    def as_dicts(self) -> list[dict[str, Any]]:
        return [i.as_dict() for i in self.intercepts]
