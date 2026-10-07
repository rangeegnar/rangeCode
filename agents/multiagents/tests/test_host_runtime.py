"""宿主运行期缺陷的回归测试 —— 零模型调用。

三个都是实测遇到的真实缺陷，各自在**当前环境里就能稳定复现或验证**：

1. `_run_shell` 的 `subprocess.run(text=True)` 没写 encoding → 中文 Windows 上用
   cp936(GBK) 解子进程的 UTF-8 输出；解码失败只在 reader 线程里抛，主线程拿到
   `stdout=None` + `returncode=0`，工具于是静默返回 `(no output)`。
   本文件的测试**强制 PYTHONUTF8=0**，所以在任何环境下都能把这条钉住。
2. `save_session` 的 `json.dumps` 没写 `ensure_ascii=False` → 中文被转义成 `\\uXXXX`，
   文件变成纯 ASCII，人打开只看到转义序列。
3. `_schedule_background_skill_task` 的 done 回调用 `except Exception` 接
   `done_task.result()`，而 `asyncio.CancelledError` 继承 `BaseException`，
   接不住 → 任务被取消时异常冒到事件循环的默认处理器。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

OPENAI_KW = {"model": "ecnu-max",
             "api_base": "https://chat.ecnu.edu.cn/open/api/v1",
             "api_key": "dummy"}

# 在子进程里调用 _run_shell 并把它返回的文本原样吐出来。
# 用 stdout.buffer 输出，避免驱动脚本自己被编码干扰。
_DRIVER = (
    "import sys\n"
    "sys.path[:0] = [p for p in sys.argv[2:] if p]\n"
    "from agents.tools import _run_shell\n"
    "out = _run_shell({'command': sys.argv[1]})\n"
    "sys.stdout.buffer.write(('<<<' + out + '>>>').encode('utf-8'))\n"
)


def _call_run_shell_in_gbk_locale(tmp_path: Path, child_source: str) -> str:
    """在 PYTHONUTF8=0（中文 Windows 的真实默认）的子进程里跑一次 _run_shell。"""
    child = tmp_path / "emit.py"
    child.write_text(child_source, encoding="utf-8")
    command = f'"{sys.executable}" "{child}"'

    env = dict(os.environ)
    env["PYTHONUTF8"] = "0"                      # 关键：关掉 UTF-8 模式，逼出 cp936
    env.pop("PYTHONIOENCODING", None)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p and Path(p).is_dir())

    proc = subprocess.run(
        [sys.executable, "-c", _DRIVER, command, *[p for p in sys.path if p and Path(p).is_dir()]],
        env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=120, cwd=str(REPO_ROOT))

    assert "<<" in proc.stdout, (proc.stdout, proc.stderr)
    return proc.stdout.split("<<<", 1)[1].rsplit(">>>", 1)[0]


def test_run_shell_decodes_utf8_output_under_gbk_locale(tmp_path):
    """命令输出中文时不能变成 (no output)。

    失败形态是 returncode=0 且 stdout=None，于是 _run_shell 静默返回
    "(no output)" —— 模型会以为命令什么都没输出。
    """
    out = _call_run_shell_in_gbk_locale(
        tmp_path,
        "import sys\nsys.stdout.buffer.write('中文输出OK'.encode('utf-8'))\n")
    assert out != "(no output)", "又变回静默空了 —— encoding 又丢了？"
    assert "中文输出OK" in out, out


def test_run_shell_survives_dirty_bytes(tmp_path):
    """偶发的非法字节不该让整条命令的输出全部丢掉。"""
    out = _call_run_shell_in_gbk_locale(
        tmp_path,
        "import sys\n"
        "sys.stdout.buffer.write(b'\\xff\\xfe head ')\n"
        "sys.stdout.buffer.write('尾部中文'.encode('utf-8'))\n")
    assert out != "(no output)", out
    assert "head" in out and "尾部中文" in out, out


def test_run_shell_is_not_the_only_place_with_explicit_encoding():
    """新加 subprocess 调用时别忘了 encoding —— 这条防止回归。"""
    from agents import tools as tools_mod

    src = Path(tools_mod.__file__).read_text(encoding="utf-8")
    calls = [i for i, line in enumerate(src.split("\n"), 1) if "subprocess.run(" in line]
    assert calls, "没找到 subprocess.run，检查是否被重写了"
    for line_no in calls:
        window = "\n".join(src.split("\n")[line_no - 1: line_no + 11])
        assert "encoding=" in window, f"第 {line_no} 行的 subprocess.run 没写 encoding"


# ── 会话文件的中文 ─────────────────────────────────────────
def test_session_file_stores_readable_chinese(tmp_path, monkeypatch):
    """会话文件里中文应以原字符存在，而不是 \\uXXXX 转义。

    数据本来就没坏（json.loads 读得出正确中文），这条防的是"人打开看不懂"。
    """
    monkeypatch.chdir(tmp_path)
    from agents.session import load_session, save_session

    payload = {"metadata": {"id": "t"}, "openaiMessages": [
        {"role": "user", "content": "把这段中文原样存下来：编码检查"}]}
    save_session("t", payload)

    path = tmp_path / ".bear" / "sessions" / "t.json"
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    assert "编码检查" in text, "中文被转义了 —— ensure_ascii 又被打回默认值？"
    assert "\\u" not in text, f"仍有转义序列：{text[:200]}"

    # 往返必须一致
    assert json.loads(text) == json.loads(json.dumps(payload, ensure_ascii=False))
    assert load_session("t")["openaiMessages"][0]["content"].endswith("编码检查")


# ── 后台任务的取消异常 ─────────────────────────────────────
def test_cancelled_background_task_does_not_escape_done_callback():
    """任务被取消时，CancelledError 不该从 done 回调冒到事件循环。

    取消发生在 REPL 退出时（asyncio.run 收尾），症状是一段
    "Exception in callback ... _done()"。
    """
    import asyncio

    from agents.agent import Agent

    async def scenario() -> list[dict]:
        agent = Agent(**OPENAI_KW)
        loop = asyncio.get_running_loop()
        seen: list[dict] = []
        loop.set_exception_handler(lambda _loop, ctx: seen.append(ctx))

        async def forever() -> None:
            await asyncio.sleep(3600)

        agent._schedule_background_skill_task(forever())
        await asyncio.sleep(0)

        pending = [t for t in agent._background_skill_tasks if not t.done()]
        assert pending, "后台任务没有被调度起来，测试前提不成立"
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await asyncio.sleep(0)          # 让 done 回调跑完
        return seen

    escaped = asyncio.run(scenario())
    assert escaped == [], f"取消异常从 done 回调逃逸到事件循环：{escaped}"


def test_real_background_failure_is_reported_not_swallowed(capsys):
    """真实失败必须被打印出来，不能被 done 回调静默吞掉。"""
    import asyncio

    from agents.agent import Agent

    async def scenario() -> None:
        agent = Agent(**OPENAI_KW)

        async def boom() -> None:
            raise RuntimeError("后台技能演化失败")

        agent._schedule_background_skill_task(boom())
        await asyncio.sleep(0)
        await asyncio.gather(*[t for t in agent._background_skill_tasks],
                             return_exceptions=True)
        await asyncio.sleep(0)

    asyncio.run(scenario())
    captured = capsys.readouterr()
    assert "后台技能演化失败" in (captured.out + captured.err), captured
