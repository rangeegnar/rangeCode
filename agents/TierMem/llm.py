"""LLM 访问层：一个可注入的协议 + 一个标准库实现。

**不含任何容灾机制。** 请求失败就抛异常，凭据缺失就报错，响应结构不对就
KeyError/IndexError。不重试、不降级、不返回空值、不做结果补齐——出错就失败，
让调用方自己决定怎么处理。

集成时（``agents/agent.py``）不需要这个模块的客户端：把宿主已有的调用包成
``AgentChatLLM`` 传进来即可。独立运行或测试时才用 ``HttpLLM``，它只用标准库
``urllib``，与本项目 ``pyproject.toml`` 的零依赖一致。
"""

from __future__ import annotations

import asyncio
import json
import urllib.request
from typing import Protocol, runtime_checkable

from . import config


@runtime_checkable
class LLMProtocol(Protocol):
    """TierMem 需要的全部模型能力。"""

    async def chat(self, system: str, user: str, *, max_tokens: int) -> str:
        """单轮对话，返回纯文本。"""
        ...

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """批量取向量，返回顺序与入参一致。"""
        ...


class HttpLLM:
    """OpenAI 兼容的 HTTP 客户端，标准库实现。

    只做两件事：``/chat/completions`` 与 ``/embeddings``。
    """

    def __init__(
        self,
        api_base: str,
        api_key: str,
        chat_model: str,
        embedding_model: str,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.chat_model = chat_model
        self.embedding_model = embedding_model

    def _post_sync(self, path: str, payload: dict) -> dict:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.api_base + path,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key,
            },
        )
        with urllib.request.urlopen(request, timeout=config.LLM_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))

    async def _post(self, path: str, payload: dict) -> dict:
        # urllib 是阻塞的，丢到线程里避免卡住事件循环
        return await asyncio.to_thread(self._post_sync, path, payload)

    async def chat(self, system: str, user: str, *, max_tokens: int) -> str:
        data = await self._post(
            "/chat/completions",
            {
                "model": self.chat_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": max_tokens,
                "temperature": 0,
            },
        )
        return data["choices"][0]["message"]["content"] or ""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        batch = config.EMBEDDING_BATCH_SIZE
        for start in range(0, len(texts), batch):
            data = await self._post(
                "/embeddings",
                {"model": self.embedding_model, "input": texts[start : start + batch]},
            )
            rows = sorted(data["data"], key=lambda row: row["index"])
            vectors.extend(row["embedding"] for row in rows)
        return vectors


class AgentChatLLM:
    """把宿主（如 ``agents/agent.py``）已有的 chat 调用包成 ``LLMProtocol``。

    chat 用宿主已配好鉴权的模型；向量必须由宿主提供一个 ``HttpLLM``，
    因为对话客户端通常没有向量能力。两者都是必填，没有默认值也没有降级。
    """

    def __init__(self, chat_fn, embed_client: HttpLLM) -> None:
        self._chat_fn = chat_fn
        self._embed_client = embed_client

    async def chat(self, system: str, user: str, *, max_tokens: int) -> str:
        return await self._chat_fn(system, user, max_tokens=max_tokens)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self._embed_client.embed(texts)


# ═══════════════════════ 模型输出解析 ═══════════════════════


def extract_json(text: str) -> dict:
    """从自由文本里抽出第一个完整 JSON 对象。

    用括号配对扫描并跳过字符串内部，比正则稳：模型常在 JSON 前后加解释，
    或者字符串里带花括号。抽不到就抛 ``ValueError``。
    """
    text = (text or "").strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("回复里没有 JSON 对象")
    depth = 0
    in_string = False
    escaped = False
    for position in range(start, len(text)):
        ch = text[position]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : position + 1])
    raise ValueError("JSON 对象括号不配对（多半是被 max_tokens 截断了）")
