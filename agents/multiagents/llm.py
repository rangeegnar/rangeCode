"""模型调用层：执行与验证两个槽位，后端可注入。

双模型是 VMAO 的核心设计（Sonnet 执行 / Opus 验证）。本项目对应：
    MODEL      = ecnu-max   执行（planner / worker / synthesizer）
    MODEL4TEST = ecnu-plus  验证（verifier / replanner）

为什么验证必须独立：MAST 的 14 条失败模式里"错误验证"占 9.1%，比"无验证"（8.2%）
更高频。同一模型既执行又验证，它的系统性盲区会在两个环节同时出现，验证退化成自我确认。

后端可注入，是为了让调度的穷举测试不需要任何模型调用 —— 上一轮那个
"4 个子任务派发 15 次"的 bug 是纯逻辑错误，用假后端就能发现，不必烧真 token。
"""
from __future__ import annotations

import json
import re
from typing import Any, Protocol

from agents import settings

from .config import DEFAULT_BUDGET, ModelConfig, load_model_config

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)

SLOT_EXEC = "exec"
SLOT_VERIFY = "verify"


class OrchestratorError(RuntimeError):
    """编排层自身出错。不吞、不降级，直接抛给调用方。"""


class LLMBackend(Protocol):
    def call_json(self, system: str, user: str, max_tokens: int | None = None,
                  *, slot: str = SLOT_EXEC) -> tuple[Any, dict[str, int]]: ...
    def call_text(self, system: str, user: str, *, slot: str = SLOT_EXEC) -> tuple[str, dict[str, int]]: ...


def extract_json(text: str) -> Any:
    """从模型输出里取出 JSON。取不到就抛错，不给默认值。"""
    stripped = text.strip()
    fenced = _FENCE_RE.match(stripped)
    cleaned = fenced.group(1) if fenced else stripped
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise OrchestratorError(
                f"无法解析模型返回的 JSON：{exc}；原文前 500 字：{cleaned[:500]}") from exc
    raise OrchestratorError(f"模型返回里找不到 JSON 对象；原文前 500 字：{cleaned[:500]}")


def is_rate_limit_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(m in text for m in (
        "429", "rate limit", "ratelimit", "model_concurrency_exceeded",
        "concurrency", "too many requests", "overloaded"))


# ── 真实后端 ──────────────────────────────────────────────
class LangChainBackend:
    """两个槽位各自一个 ChatOpenAI 客户端，都指向同一个自建端点。

    `config` 是 property，每次都从模块级的 `model_config()` 现取，不在构造时快照。
    否则宿主注入的执行模型（`apply_host_model`）在 backend 已经被创建过的情况下
    不会生效 —— 那是"同一个进程里第二次跑还用的是旧模型"这类难查的问题。
    """

    def __init__(self, config: ModelConfig | None = None):
        self._explicit_config = config
        # slot -> (model, base_url, client)；配置变了就重建，不会用到过期客户端
        self._clients: dict[str, tuple[str, str, Any]] = {}

    @property
    def config(self) -> ModelConfig:
        return self._explicit_config or model_config()

    @property
    def independent(self) -> bool:
        return self.config.verify_is_independent

    def _client(self, slot: str) -> Any:
        cfg = self.config
        model = cfg.exec_model if slot == SLOT_EXEC else cfg.verify_model
        cached = self._clients.get(slot)
        if cached is None or cached[0] != model or cached[1] != cfg.openai_base:
            from langchain_openai import ChatOpenAI

            cached = (model, cfg.openai_base, ChatOpenAI(
                model=model,
                api_key=cfg.api_key,
                base_url=cfg.openai_base,
                temperature=0.0,
                max_retries=DEFAULT_BUDGET.llm_max_retries,
                timeout=DEFAULT_BUDGET.llm_timeout_s,
            ))
            self._clients[slot] = cached
        return cached[2]

    @staticmethod
    def _usage(msg: Any) -> dict[str, int]:
        usage = getattr(msg, "usage_metadata", None) or {}
        return {
            "input": int(usage.get("input_tokens", 0) or 0),
            "output": int(usage.get("output_tokens", 0) or 0),
        }

    def _slot_max_output(self, slot: str) -> int:
        """该槽位模型的单次输出上限。

        与主流程共用同一处解析（`agents/settings.py`）：先看 .env 里
        按角色配的 MODEL_MAX_OUTPUT / MODEL4TEST_MAX_OUTPUT，再看按角色默认值，最后通用兜底。
        这样编排层和执行层的输出上限不会各说各话。
        """
        model = self.config.exec_model if slot == SLOT_EXEC else self.config.verify_model
        return settings.max_output_tokens(model)

    def _invoke(self, slot: str, system: str, user: str,
                max_tokens: int | None) -> tuple[str, dict[str, int]]:
        from langchain_core.messages import HumanMessage, SystemMessage

        client = self._client(slot)
        # 调用方没指定就用该槽位的模型上限（原来是什么都不发，由服务端决定）
        limit = max_tokens if max_tokens else self._slot_max_output(slot)
        bound = client.bind(max_tokens=limit) if limit else client
        msg = bound.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        text = msg.content if isinstance(msg.content, str) else str(msg.content)
        return text, self._usage(msg)

    def call_json(self, system: str, user: str, max_tokens: int | None = None,
                  *, slot: str = SLOT_EXEC) -> tuple[Any, dict[str, int]]:
        text, tokens = self._invoke(slot, system, user, max_tokens)
        return extract_json(text), tokens

    def call_text(self, system: str, user: str, *, slot: str = SLOT_EXEC) -> tuple[str, dict[str, int]]:
        return self._invoke(slot, system, user, None)


# ── 后端单例 ──────────────────────────────────────────────
_backend: LLMBackend | None = None
_config: ModelConfig | None = None


def set_backend(backend: LLMBackend | None) -> None:
    """注入后端。测试传假后端；生产传 None 走真实后端。"""
    global _backend
    _backend = backend


def backend() -> LLMBackend:
    global _backend
    if _backend is None:
        _backend = LangChainBackend()
    return _backend


def model_config() -> ModelConfig:
    global _config
    if _config is None:
        _config = load_model_config()
    return _config


def set_model_config(cfg: ModelConfig | None) -> None:
    """注入模型配置。传 None 表示回到读 .env。"""
    global _config
    _config = cfg


def apply_host_model(*, host_model: str | None = None, host_api_base: str | None = None,
                     host_api_key: str | None = None,
                     host_embedding_model: str | None = None) -> ModelConfig:
    """让执行槽位跟随宿主的模型配置，验证槽位仍来自 .env 的 MODEL4TEST。

    为什么必须跟随宿主：`agents/main.py` 把解析后的 model / api_base / api_key 注入
    Agent，且 CLI 可以用 `--model` 覆盖。编排层如果自己去读 .env 的 MODEL，用户在
    CLI 换模型时两边就不一致了 —— 表现是"我明明换了模型，编排还在用原来那个"。

    验证模型不给宿主覆盖，是因为"执行与验证用不同模型"是这套设计的前提
    （VMAO 的核心，MAST 里"错误验证"占 9.1%）。宿主只有一个 model，
    让它覆盖验证槽位等于把双模型退化成单模型。

    任何宿主参数都没给时，行为与直接读 .env 完全一致。
    """
    if not any((host_model, host_api_base, host_api_key, host_embedding_model)):
        return model_config()

    base = model_config()
    cfg = ModelConfig(
        api_key=host_api_key or base.api_key,
        api_base=host_api_base or base.api_base,
        exec_model=host_model or base.exec_model,
        verify_model=base.verify_model,
        verify_is_independent=base.verify_is_independent,
        embedding_model=host_embedding_model or base.embedding_model,
    )
    set_model_config(cfg)
    return cfg


def reset_model_config() -> None:
    """清掉注入的配置。测试用。"""
    set_model_config(None)


# ── 用量计数 ──────────────────────────────────────────────
# 为什么在这里数：编排层节点（plan / verify / replan / synthesize）和
# `_engine_plain` 的 worker 全都走本模块的 `call_json` / `call_text`，
# 所以这里能拿到它们的精确 input/output 拆分。
# 唯一不经这里的是 `lg-tool-agent` 的 worker —— 它由 create_agent 内部自己
# 建 ChatOpenAI 调模型，那部分 token 从 SubResult 上取（见 interface.py）。
#
# 有了精确拆分才能如实回灌宿主的 total_input_tokens / total_output_tokens：
# `Agent._get_current_cost_usd` 按 input $3/M、output $15/M 计价，混着算会失真。
_USAGE: dict[str, int] = {"input": 0, "output": 0}


def reset_usage() -> None:
    _USAGE["input"] = 0
    _USAGE["output"] = 0


def usage_snapshot() -> dict[str, int]:
    return dict(_USAGE)


def _account(tokens: dict[str, int]) -> dict[str, int]:
    _USAGE["input"] += int(tokens.get("input", 0) or 0)
    _USAGE["output"] += int(tokens.get("output", 0) or 0)
    return tokens


def verify_is_independent() -> bool:
    """验证器是否真的用了与执行器不同的模型。

    这条要能被断言，不能只看配置 —— 运行记录里会带上它。
    """
    return bool(getattr(backend(), "independent", False))


def call_json(system: str, user: str, max_tokens: int | None = None, *, slot: str = SLOT_EXEC):
    payload, tokens = backend().call_json(system, user, max_tokens, slot=slot)
    _account(tokens)
    return payload, tokens


def call_text(system: str, user: str, *, slot: str = SLOT_EXEC):
    text, tokens = backend().call_text(system, user, slot=slot)
    _account(tokens)
    return text, tokens
