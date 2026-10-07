"""配置解析测试 —— 重点是双模型，以及防"向量模型被当成聊天模型"。

这条不是假想：项目历史上 VECTOR_MODEL 那行带行尾注释，dotenv 把整行（含注释）
塞进 os.environ，结果注释被当成模型名发出去、接口 401/400。所以：
    1. 所有取值都必须先剥 `#` 之后的内容再 strip
    2. MODEL4TEST 才是验证模型，VECTOR_MODEL 只给 TierMem 用，不能被当成聊天模型
"""
from __future__ import annotations

import pytest

from agents.multiagents.config import _clean_env, load_model_config

BASE = {
    "APIKEY": "sk-test",
    "API": "https://example.invalid/open/api/v1",
    "MODEL": "ecnu-max",
}


def _set(monkeypatch, **env):
    """设置环境变量，并把 .env 加载打桩掉。

    必须打桩：load_model_config() 内部会调 load_dotenv 重读真实的 .env，
    monkeypatch 删掉的键会被它灌回来，导致"缺键"这类用例根本测不出来。
    """
    monkeypatch.setattr("agents.multiagents.config._load_env", lambda: None)
    for key in ("APIKEY", "API", "MODEL", "MODEL4TEST", "VECTOR_MODEL",
                "OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    for key, value in {**BASE, **env}.items():
        if value is not None:
            monkeypatch.setenv(key, value)


# ── _clean_env ────────────────────────────────────────────

def test_clean_env_strips_trailing_comment_and_space():
    assert _clean_env("ecnu-embedding-small # 1024 dimension") == "ecnu-embedding-small"
    assert _clean_env("ecnu-embedding-small ") == "ecnu-embedding-small"
    assert _clean_env("  ecnu-plus  ") == "ecnu-plus"
    assert _clean_env("") is None
    assert _clean_env(None) is None
    assert _clean_env("   ") is None


# ── 双模型 ────────────────────────────────────────────────

def test_model4test_is_the_verify_model(monkeypatch):
    _set(monkeypatch, MODEL4TEST="ecnu-plus", VECTOR_MODEL="ecnu-embedding-small")
    cfg = load_model_config()
    assert cfg.exec_model == "ecnu-max"
    assert cfg.verify_model == "ecnu-plus"
    assert cfg.verify_is_independent is True
    assert cfg.embedding_model == "ecnu-embedding-small"


def test_missing_model4test_falls_back_and_marks_not_independent(monkeypatch):
    """缺 MODEL4TEST 时回落执行模型，但必须把"不独立"标出来，不能假装独立。"""
    _set(monkeypatch, VECTOR_MODEL="ecnu-embedding-small")
    cfg = load_model_config()
    assert cfg.verify_model == "ecnu-max"
    assert cfg.verify_is_independent is False


def test_model4test_equal_to_model_is_not_independent(monkeypatch):
    _set(monkeypatch, MODEL4TEST="ecnu-max")
    cfg = load_model_config()
    assert cfg.verify_is_independent is False


def test_vector_model_never_becomes_a_chat_model(monkeypatch):
    """防混淆：VECTOR_MODEL 不能被当成执行或验证模型。"""
    _set(monkeypatch, MODEL4TEST=None, VECTOR_MODEL="ecnu-embedding-small")
    cfg = load_model_config()
    assert cfg.verify_model != "ecnu-embedding-small"
    assert cfg.exec_model != "ecnu-embedding-small"


def test_verify_model_equal_to_vector_model_raises(monkeypatch):
    """如果有人把向量模型名串到了 MODEL4TEST，必须直接报错而不是照跑。"""
    _set(monkeypatch, MODEL4TEST="ecnu-embedding-small", VECTOR_MODEL="ecnu-embedding-small")
    with pytest.raises(RuntimeError) as exc:
        load_model_config()
    assert "向量模型" in str(exc.value)


def test_trailing_comment_on_model_fields_is_stripped(monkeypatch):
    """模型字段带行尾注释时不能把注释当模型名发出去。"""
    _set(monkeypatch, MODEL="ecnu-max # 主力", MODEL4TEST="ecnu-plus # 验证")
    cfg = load_model_config()
    assert cfg.exec_model == "ecnu-max"
    assert cfg.verify_model == "ecnu-plus"


def test_missing_required_keys_raise(monkeypatch):
    _set(monkeypatch)
    monkeypatch.delenv("API", raising=False)
    with pytest.raises(RuntimeError) as exc:
        load_model_config()
    assert "API" in str(exc.value)


def test_openai_base_has_no_trailing_slash(monkeypatch):
    _set(monkeypatch, API="https://example.invalid/open/api/v1/")
    cfg = load_model_config()
    assert cfg.openai_base == "https://example.invalid/open/api/v1"
