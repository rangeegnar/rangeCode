from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

_HERE = Path(__file__).resolve()
PKG_DIR = _HERE.parent              # .../agents/multiagents
PROJECT_ROOT = _HERE.parents[2]     # .../RangeCode


@dataclass(frozen=True)
class Budget:
    """编排预算。

    VMAO 默认值（arXiv:2603.11445 Table 5）：
        max_iterations=3, token_budget=1M, ready_threshold=0.8,
        high_confidence=0.75, diminishing_returns=0.05, max_concurrent=3,
        agent_timeout=600s
    另有安全阀：同工具连续调用上限 10 次，单 agent 总工具调用上限 50 次。
    """

    ready_threshold: float = 0.8
    high_confidence: float = 0.75
    high_confidence_coverage: float = 0.5
    diminishing_returns: float = 0.05
    hierarchical_synth_chars: int = 15_000
    hierarchical_synth_results: int = 10
    max_concurrent: int = 2
    max_iterations: int = 2
    token_budget: int = 200_000
    # VMAO 是 600s。调试期间放宽到 18000s（5 小时）
    agent_timeout_s: int = 600
    # 工具调用上限
    worker_recursion_limit: int = 50
    same_tool_limit: int = 10

    tau_len: int = 3000
    tau_loop: int = 5
    tau_step: int = 8
    # 净化默认开：论文消融显示去掉它节省从 29.68% 掉到 15.96%
    enable_purification: bool = True
    # 纠错与引导默认关：论文承认过度干预伤性能，且"噪声即信号"意味着净化本身有风险
    enable_guidance: bool = False
    enable_verification_action: bool = False

    # ── 图执行 ────────────────────────────────────────────────
    # recursion_limit 数的是超步。3 波次 + 2 轮验证约需 12-16 超步
    graph_recursion_limit: int = 50
    llm_max_retries: int = 8
    # 单次 HTTP 请求,调试期间放宽到 18000s（5 小时）。
    llm_timeout_s: int = 180


DEFAULT_BUDGET = Budget()


def with_overrides(budget: Budget, **kwargs) -> Budget:
    return replace(budget, **{k: v for k, v in kwargs.items() if v is not None})


from agents.settings import clean_env as _clean_env


def _load_env() -> None:
    candidate = PROJECT_ROOT / ".env"
    if candidate.is_file():
        load_dotenv(candidate, override=False)


@dataclass(frozen=True)
class ModelConfig:
    api_key: str
    api_base: str
    exec_model: str
    verify_model: str
    verify_is_independent: bool
    embedding_model: str | None = None

    @property
    def openai_base(self) -> str:
        return self.api_base.rstrip("/")


def load_model_config() -> ModelConfig:
    """解析模型配置。缺项直接抛错，不设默认端点兜底。"""
    _load_env()

    api_key = _clean_env(os.environ.get("APIKEY")) or _clean_env(os.environ.get("OPENAI_API_KEY"))
    api_base = (
        _clean_env(os.environ.get("API"))
        or _clean_env(os.environ.get("OPENAI_BASE_URL"))
        or _clean_env(os.environ.get("ANTHROPIC_BASE_URL"))
    )
    exec_model = _clean_env(os.environ.get("MODEL"))
    verify_model = _clean_env(os.environ.get("MODEL4TEST"))
    embedding = _clean_env(os.environ.get("VECTOR_MODEL"))

    missing = [n for n, v in (("APIKEY", api_key), ("API", api_base), ("MODEL", exec_model)) if not v]
    if missing:
        raise RuntimeError(
            f"缺少模型配置 {missing}；请在 {PROJECT_ROOT / '.env'} 里提供。"
        )

    independent = bool(verify_model) and verify_model != exec_model
    resolved_verify = verify_model or exec_model

    # 防混淆硬断言：向量模型不是聊天模型，不能被当成验证模型
    if embedding and resolved_verify == embedding:
        raise RuntimeError(
            f"验证模型被解析成了向量模型 {embedding!r}。"
            "MODEL4TEST 才是验证模型，VECTOR_MODEL 只给 TierMem 用。"
            f"请检查 {PROJECT_ROOT / '.env'} 是否把注释或向量模型名串到了模型字段上。"
        )

    return ModelConfig(
        api_key=api_key,
        api_base=api_base,
        exec_model=exec_model,
        verify_model=resolved_verify,
        verify_is_independent=independent,
        embedding_model=embedding,
    )
