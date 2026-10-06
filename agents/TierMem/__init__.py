"""TierMem：带溯源链接的两级 Agent 记忆。

设计来源：*From Lossy to Verified: A Provenance-Aware Tiered Memory for Agents*
（arXiv:2602.17913）。核心是把检索当成一次「推理时的证据分配」问题：
默认用最便宜的摘要回答，只有在摘要缺少决定性约束时才升级去读不可变的原始日志，
并把升级中验证过的结论带着溯源写回。

模块分工：

===============  ==========================================================
``config``       全部常量与可调参数（除凭据与模型名外均为字面量，不读环境变量）
``prompts``      全部提示词模板
``store``        Tier-2 不可变分页原始日志，无 LLM 依赖
``index``        Tier-1 摘要索引 + 溯源链接 + 向量检索，无 LLM 依赖
``llm``          可注入的模型协议 + 标准库 HTTP 实现 + JSON 抽取（不含任何容灾）
``router``       充分性路由（ANSWER / ESCALATE）与评测判官
``writeback``    确定性溯源链接 + SKIP/UPDATE/ADD 裁决
``tiermem``      编排器：ingest / query / commit
===============  ==========================================================

最小用法（独立运行，自带标准库 HTTP 客户端）::

    from agents.TierMem import TierMem

    memory = TierMem()
    await memory.ingest(messages, session="s1")
    trace = await memory.query("限流阈值设多少？")
    await memory.commit("限流阈值设多少？", trace)

接入 ``agents/agent.py``（复用已有客户端，不重复造）::

    class AgentLLM:                     # 把 Agent 的客户端包成 LLMProtocol
        async def chat(self, system, user, *, max_tokens):
            return await side_query(system, user)
        async def embed(self, texts):
            return await embed_client(texts)

    memory = TierMem(llm=AgentLLM())
"""

from __future__ import annotations

from . import config, prompts
from .index import Tier1Index, Unit, cosine
from .llm import AgentChatLLM, HttpLLM, LLMProtocol, extract_json
from .router import judge, sufficiency_router
from .store import Tier2Store, est_tokens, format_page_id
from .tiermem import TierMem
from .writeback import deterministic_link, parse_facts, tokens_of, writeback

__all__ = [
    # 门面
    "TierMem",
    # 子模块
    "config",
    "prompts",
    "Tier1Index",
    "Tier2Store",
    "Unit",
    # 模型层
    "LLMProtocol",
    "HttpLLM",
    "AgentChatLLM",
    "extract_json",
    # 路由与写回
    "sufficiency_router",
    "judge",
    "writeback",
    "deterministic_link",
    "parse_facts",
    "tokens_of",
    # 工具
    "cosine",
    "est_tokens",
    "format_page_id",
]
