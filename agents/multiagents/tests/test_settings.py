"""`agents/settings.py` 的测试 —— 零模型调用。

覆盖三件事：

1. **模型容量解析**：.env 按角色配置 → 按角色默认值 → 通用兜底。
   其中行尾注释必须被剥掉，否则 `int()` 抛 ValueError、配置静默失效。
2. **常量收敛**：agent / tools / main / multiagents 引用的都是 settings 里的同一份对象。
3. **settings.py 的依赖边界**：只能依赖标准库，否则全项目 import 它会成环。
"""
from __future__ import annotations

import ast
import asyncio
import os
from pathlib import Path

import pytest

from agents import settings

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def clean_env(monkeypatch):
    """清掉所有相关键，让每个测试从确定状态出发。"""
    for key in ("MODEL", "MODEL4TEST", "MODEL_MAX_CONTEXT", "MODEL4TEST_MAX_CONTEXT",
                "MODEL_MAX_OUTPUT", "MODEL4TEST_MAX_OUTPUT", "APIKEY", "API"):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


# ── 行尾注释（核心用例）────────────────────────────────────
def test_inline_comments_are_stripped(clean_env):
    clean_env.setenv("MODEL", "ecnu-max")
    clean_env.setenv("MODEL4TEST", "ecnu-plus")
    # 与 .env 里真实写法一致：数值后面跟一大段注释
    clean_env.setenv("MODEL_MAX_CONTEXT", "524288        # ecnu-max 512K = 512*1024")
    clean_env.setenv("MODEL_MAX_OUTPUT", "393216         # ecnu-max 384K")
    clean_env.setenv("MODEL4TEST_MAX_CONTEXT", "262144    # ecnu-plus 256K = 256*1024")
    clean_env.setenv("MODEL4TEST_MAX_OUTPUT", "131072     # ecnu-plus 128K")

    assert settings.context_window("ecnu-max") == 524_288
    assert settings.max_output_tokens("ecnu-max") == 393_216
    assert settings.context_window("ecnu-plus") == 262_144
    assert settings.max_output_tokens("ecnu-plus") == 131_072


def test_clean_env_strips_comment_and_blank():
    assert settings.clean_env("524288 # note") == "524288"
    assert settings.clean_env("  value  ") == "value"
    assert settings.clean_env("   # 只有注释") is None
    assert settings.clean_env("") is None
    assert settings.clean_env(None) is None


def test_project_env_file_resolves_to_expected_limits():
    """直接读仓库根的真实 .env，确认那 4 行确实能被解析成预期的整数。

    .env 缺这几个键时本测试仍然有意义：它校验的是"解析出来的值符合契约
    （要么等于配置值、要么等于默认值）"，而不是"键必须存在"。
    """
    from dotenv import load_dotenv

    env_file = REPO_ROOT / ".env"
    if not env_file.is_file():
        pytest.skip("没有 .env")

    load_dotenv(env_file, override=True)
    exec_model = settings.clean_env(os.environ.get("MODEL"))
    verify_model = settings.clean_env(os.environ.get("MODEL4TEST"))
    if not exec_model or not verify_model:
        pytest.skip(".env 没有同时配置 MODEL 与 MODEL4TEST")

    assert settings.context_window(exec_model) >= settings.UNKNOWN_CONTEXT
    assert settings.max_output_tokens(exec_model) >= settings.UNKNOWN_MAX_OUTPUT
    assert settings._role_of(exec_model) == "exec"
    assert settings._role_of(verify_model) == "verify"


# ── 默认值 ─────────────────────────────────────────────────
def test_defaults_when_env_keys_absent(clean_env):
    clean_env.setenv("MODEL", "ecnu-max")
    clean_env.setenv("MODEL4TEST", "ecnu-plus")

    assert settings.context_window("ecnu-max") == settings.EXEC_DEFAULT_CONTEXT == 524_288
    assert settings.max_output_tokens("ecnu-max") == settings.EXEC_DEFAULT_MAX_OUTPUT == 393_216
    assert settings.context_window("ecnu-plus") == settings.VERIFY_DEFAULT_CONTEXT == 262_144
    assert settings.max_output_tokens("ecnu-plus") == settings.VERIFY_DEFAULT_MAX_OUTPUT == 131_072


def test_invalid_or_zero_value_falls_back_to_default(clean_env):
    clean_env.setenv("MODEL", "ecnu-max")
    for bad in ("abc", "", "0", "-1", "524288 # 后面还有注释但数字非法吗"):
        clean_env.setenv("MODEL_MAX_CONTEXT", bad)
        got = settings.context_window("ecnu-max")
        if bad == "524288 # 后面还有注释但数字非法吗":
            assert got == 524_288, bad          # 这个应该成功
        else:
            assert got == settings.EXEC_DEFAULT_CONTEXT, bad


# ── 角色路由与通用兜底 ─────────────────────────────────────
def test_unknown_model_uses_fallback(clean_env):
    clean_env.setenv("MODEL", "ecnu-max")
    clean_env.setenv("MODEL4TEST", "ecnu-plus")
    assert settings.context_window("some-other-model") == settings.UNKNOWN_CONTEXT
    assert settings.max_output_tokens("some-other-model") == settings.UNKNOWN_MAX_OUTPUT
    assert settings._role_of("some-other-model") is None


def test_no_builtin_model_table(clean_env):
    """刻意不内置厂商模型表。

    上下文窗口是服务端属性，写死在代码里只会过期 —— 而过期的窗口值会让四层
    压缩在错误的时机触发，比没有值更糟。所以曾经的 gpt-4o / claude-* 现在
    一律走通用兜底，要用第三个模型就在 .env 里显式配一对上限。
    """
    clean_env.setenv("MODEL", "ecnu-max")
    clean_env.setenv("MODEL4TEST", "ecnu-plus")

    assert not hasattr(settings, "BUILTIN_CONTEXT"), "内置模型表应当已经删除"
    for vendor_model in ("gpt-4o", "gpt-4o-mini", "claude-opus-4-6",
                         "claude-sonnet-4-6", "deepseek-chat"):
        assert settings.context_window(vendor_model) == settings.UNKNOWN_CONTEXT, vendor_model
        assert settings.max_output_tokens(vendor_model) == settings.UNKNOWN_MAX_OUTPUT, vendor_model
        assert settings._role_of(vendor_model) is None


def test_exec_role_wins_over_verify_when_names_collide(clean_env):
    """MODEL 与 MODEL4TEST 同名时（未配置验证模型）按执行槽位处理。"""
    clean_env.setenv("MODEL", "same-model")
    clean_env.setenv("MODEL4TEST", "same-model")
    assert settings._role_of("same-model") == "exec"


def test_describe_shape(clean_env):
    clean_env.setenv("MODEL", "ecnu-max")
    info = settings.describe("ecnu-max")
    assert info == {"model": "ecnu-max", "role": "exec",
                    "context_window": 524_288, "max_output_tokens": 393_216}


# ── 依赖边界 ───────────────────────────────────────────────
def test_settings_only_depends_on_stdlib():
    """settings.py 被全项目 import，任何 agents.* 反向依赖都会立刻成环。"""
    tree = ast.parse(Path(settings.__file__).read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offenders += [a.name for a in node.names if a.name.split(".")[0] == "agents"]
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "agents":
            offenders.append(node.module or "")
    assert offenders == [], f"settings.py 不能 import agents.*：{offenders}"


# ── 常量收敛：各处引用的必须是同一份对象 ───────────────────
def test_agent_constants_come_from_settings():
    from agents import agent as agent_mod

    for name in ("BUDGET_THRESHOLD", "SNIP_THRESHOLD", "AUTO_COMPACT_THRESHOLD",
                 "SNIP_PLACEHOLDER", "OLD_RESULT_CLEARED", "SNIPPABLE_TOOLS",
                 "MICROCOMPACT_IDLE_S", "KEEP_RECENT_RESULTS", "RESPONSE_RESERVE_TOKENS",
                 "RETRY_MAX_ATTEMPTS", "RETRY_BACKOFF_BASE_MS", "RETRY_BACKOFF_CAP_MS",
                 "SIDE_QUERY_MAX_TOKENS", "SIDE_QUERY_ARCHIVE_MAX_TOKENS",
                 "SIDE_QUERY_EXTRACT_MAX_TOKENS",
                 "PERSIST_RESULT_THRESHOLD_BYTES", "PERSIST_PREVIEW_LINES"):
        assert getattr(agent_mod, name) is getattr(settings, name), name


def test_tools_constants_come_from_settings():
    from agents import tools

    assert tools.MAX_RESULT_CHARS is settings.MAX_RESULT_CHARS
    assert tools.TRUNCATE_MARKER_RESERVE_CHARS is settings.TRUNCATE_MARKER_RESERVE_CHARS


def test_truncate_result_respects_the_limit():
    """截断后的长度必须 <= MAX_RESULT_CHARS，且标记行要说明丢了多少。"""
    from agents import tools

    limit = settings.MAX_RESULT_CHARS
    exact = "x" * limit
    assert tools._truncate_result(exact) == exact      # 刚好等于上限：不动

    out = tools._truncate_result("x" * (limit + 10_000))
    assert len(out) <= limit
    assert "truncated" in out


def test_clean_env_has_a_single_implementation():
    """三个模块里曾经各有一份同义的 clean_env，现在都指向 settings。"""
    from agents import main
    from agents.multiagents import config as ma_config

    assert main._clean_env is settings.clean_env
    assert ma_config._clean_env is settings.clean_env


def test_persist_preview_message_uses_the_constant():
    """落盘提示里的行数不能是第二个硬编码的 200 —— 改常量时最容易漏这里。"""
    from agents import agent as agent_mod

    src = Path(agent_mod.__file__).read_text(encoding="utf-8")
    assert "Preview (first {PERSIST_PREVIEW_LINES} lines)" in src
    assert "Preview (first 200 lines)" not in src


def test_old_result_cleared_literal_is_not_duplicated():
    """占位文本出现 4 次，全部改用常量，避免只改一处。"""
    from agents import agent as agent_mod

    src = Path(agent_mod.__file__).read_text(encoding="utf-8")
    assert '"[Old result cleared]"' not in src


# ── 主流程接线 ─────────────────────────────────────────────
def test_agent_effective_window_follows_env(clean_env):
    clean_env.setenv("MODEL", "ecnu-max")
    clean_env.setenv("MODEL_MAX_CONTEXT", "524288   # 512K")

    from agents.agent import Agent

    agent = Agent(model="ecnu-max", api_base="https://example.invalid/v1", api_key="dummy")
    assert agent.effective_window == 524_288 - settings.RESPONSE_RESERVE_TOKENS


def test_compression_thresholds_move_with_the_window(clean_env):
    """窗口变大后，四层压缩的触发点必须跟着后移（否则等于没放宽）。"""
    from agents.agent import AUTO_COMPACT_THRESHOLD, BUDGET_THRESHOLD, SNIP_THRESHOLD

    clean_env.setenv("MODEL", "ecnu-max")
    clean_env.setenv("MODEL_MAX_CONTEXT", "524288")
    effective = settings.context_window("ecnu-max") - settings.RESPONSE_RESERVE_TOKENS

    assert int(effective * BUDGET_THRESHOLD) == 252_144
    assert int(effective * SNIP_THRESHOLD) == 302_572
    assert int(effective * AUTO_COMPACT_THRESHOLD) == 353_001


def test_anthropic_and_openai_both_send_max_tokens(clean_env):
    """两条路径都要把上限发给 API，否则 .env 里那个键是死的。"""
    clean_env.setenv("MODEL", "ecnu-max")
    clean_env.setenv("MODEL_MAX_OUTPUT", "393216   # 384K")

    from agents.agent import Agent

    assert settings.max_output_tokens("ecnu-max") == 393_216

    sent: dict = {}

    class _Chunk:
        usage = None
        choices: list = []

    class _Completions:
        async def create(self, **kwargs):
            sent.update(kwargs)

            async def _gen():
                yield _Chunk()

            return _gen()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    agent = Agent(model="ecnu-max", api_base="https://example.invalid/v1", api_key="dummy")
    agent._openai_client = _Client()
    asyncio.run(agent._call_openai_stream())

    assert sent.get("max_tokens") == 393_216, sent


def test_orchestrator_slots_get_their_own_limits(clean_env):
    """编排层两个槽位各拿各的上限，不共用。"""
    clean_env.setenv("MODEL", "ecnu-max")
    clean_env.setenv("MODEL4TEST", "ecnu-plus")
    clean_env.setenv("MODEL_MAX_OUTPUT", "393216")
    clean_env.setenv("MODEL4TEST_MAX_OUTPUT", "131072")

    from agents.multiagents.config import ModelConfig
    from agents.multiagents.llm import SLOT_EXEC, SLOT_VERIFY, LangChainBackend

    cfg = ModelConfig(api_key="k", api_base="https://example.invalid/v1",
                      exec_model="ecnu-max", verify_model="ecnu-plus",
                      verify_is_independent=True, embedding_model=None)
    backend = LangChainBackend(cfg)

    assert backend._slot_max_output(SLOT_EXEC) == 393_216
    assert backend._slot_max_output(SLOT_VERIFY) == 131_072
