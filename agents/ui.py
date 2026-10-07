"""Terminal UI rendering — colored output, spinner, tool display in RANGE design language."""

from __future__ import annotations

import sys
import threading
import time

from rich import box
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console(highlight=False)

# ─── Color Palette & Theme Tokens ─────────────────────────
COLOR_ACCENT = "#00e5ff"       # 电青色 (Electric Cyan)
COLOR_SECONDARY = "#705df2"    # 能量紫 (Deep Indigo / Violet)
COLOR_SUCCESS = "#73daca"      # 终端流光绿 (Mint / Green)
COLOR_WARN = "#e0af68"         # 预警琥珀橙 (Amber)
COLOR_ERROR = "#f7768e"        # 错误绯红 (Crimson)
COLOR_DIM = "#565f89"          # 弱化注释灰 (Dim Slate)
COLOR_TEXT = "#c0caf5"         # 主体亮文本 (Foreground White-Blue)


# ─── Basic output ──────────────────────────────────────────


def _safe_text(value: object) -> str:
    return str(value).encode("utf-8", errors="replace").decode("utf-8")


def _safe_stdout_write(text: object) -> None:
    sys.stdout.write(_safe_text(text))
    sys.stdout.flush()


RANGE_BANNER = r"""
 █▀▀█ █▀▀█ █▀▀▄ █▀▀▀ █▀▀ 
 █▄▄▀ █▄▄█ █  █ █ ▀█ █▀▀ 
 ▀ ▀▀ ▀  ▀ ▀  ▀ ▀▀▀▀ ▀▀▀ 
"""


def print_welcome() -> None:
    title = Text("R A N G E   C O D E", style=f"bold {COLOR_ACCENT}")
    subtitle = Text("Autonomous Evolvable Coding Runtime", style=f"dim {COLOR_TEXT}")
    banner = Text(RANGE_BANNER, style=f"bold {COLOR_SECONDARY}")

    commands = Table.grid(padding=(0, 2))
    commands.add_column(style=f"bold {COLOR_ACCENT}", no_wrap=True)
    commands.add_column(style=COLOR_DIM)
    commands.add_row("/plan", "Strategic read-only planning loop")
    commands.add_row("/skills", "Inspect installed agent capabilities")
    commands.add_row("/skill-create", "Synthesize a new autonomous skill")
    commands.add_row("/skill-stats", "Inspect capability evolution metrics")
    commands.add_row("/memory", "Access episodic & semantic memory matrix")
    commands.add_row("/compact", "Prune & compress context window")
    commands.add_row("exit", "Terminate active session")

    body = Table.grid()
    body.add_row(Align.center(banner))
    body.add_row(Align.center(title))
    body.add_row(Align.center(subtitle))
    body.add_row("")
    body.add_row(
        Panel(
            commands,
            title=f"[{COLOR_ACCENT}]System Operations[/{COLOR_ACCENT}]",
            border_style=COLOR_SECONDARY,
            box=box.ROUNDED,
        )
    )

    console.print()
    console.print(
        Panel(
            body,
            title=f"[bold {COLOR_ACCENT}] RANGE CORE :: READY [/bold {COLOR_ACCENT}]",
            subtitle=f"[{COLOR_DIM}]Awaiting directives[/{COLOR_DIM}]",
            border_style=COLOR_ACCENT,
            box=box.ROUNDED,
            padding=(1, 3),
        )
    )
    console.print()


def print_user_prompt() -> None:
    console.print(
        f"\n[bold {COLOR_SECONDARY}]RANGE[/bold {COLOR_SECONDARY}]"
        f"[dim] :: [/dim]"
        f"[bold {COLOR_ACCENT}]agent[/bold {COLOR_ACCENT}] "
        f"[{COLOR_DIM}]❯[/{COLOR_DIM}] ",
        end="",
    )


def print_assistant_text(text: str) -> None:
    _safe_stdout_write(text)


def print_tool_call(name: str, inp: dict) -> None:
    icon = _get_tool_icon(name)
    summary = _get_tool_summary(name, inp)
    table = Table.grid(padding=(0, 1))
    table.add_column(style=f"bold {COLOR_ACCENT}", no_wrap=True)
    table.add_column(style=COLOR_TEXT)
    table.add_row("routine", f"{icon}  {name}")
    if summary:
        table.add_row("args", _safe_text(summary))
    console.print(
        Panel(
            table,
            title=f"[bold {COLOR_ACCENT}]Execute :: {name}[/bold {COLOR_ACCENT}]",
            border_style=COLOR_SECONDARY,
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )


def print_tool_result(name: str, result: str) -> None:
    result = _safe_text(result)
    if (name in ("edit_file", "write_file")) and not result.startswith("Error"):
        _print_file_change_result(name, result)
        return
    max_len = 500
    truncated = result
    if len(result) > max_len:
        truncated = result[:max_len] + f"\n  ... ({len(result)} chars total)"
    console.print(
        Panel(
            _safe_text(truncated),
            title=f"[{COLOR_DIM}]Output :: {name}[/{COLOR_DIM}]",
            border_style=COLOR_DIM,
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )


def _print_file_change_result(_name: str, result: str) -> None:
    lines = result.split("\n")
    console.print(
        Panel(
            _safe_text(lines[0]),
            title=f"[bold {COLOR_SUCCESS}]Patch Applied[/bold {COLOR_SUCCESS}]",
            border_style=COLOR_SUCCESS,
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )

    max_display = 40
    content_lines = lines[1:]
    display_lines = content_lines[:max_display]

    for line in display_lines:
        if not line.strip():
            continue
        if line.startswith("@@"):
            console.print(f"[{COLOR_ACCENT}]  {line}[/{COLOR_ACCENT}]")
        elif line.startswith("- "):
            console.print(f"[{COLOR_ERROR}]  {line}[/{COLOR_ERROR}]")
        elif line.startswith("+ "):
            console.print(f"[{COLOR_SUCCESS}]  {line}[/{COLOR_SUCCESS}]")
        else:
            console.print(f"[{COLOR_DIM}]  {line}[/{COLOR_DIM}]")
    if len(content_lines) > max_display:
        console.print(f"[{COLOR_DIM}]  ... ({len(content_lines) - max_display} more lines)[/{COLOR_DIM}]")


def print_error(msg: str) -> None:
    console.print(
        Panel(
            _safe_text(msg),
            title=f"[bold {COLOR_ERROR}]Execution Fault[/bold {COLOR_ERROR}]",
            border_style=COLOR_ERROR,
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )


def print_confirmation(command: str) -> None:
    console.print(
        Panel(
            _safe_text(command),
            title=f"[bold {COLOR_WARN}]Authorization Required[/bold {COLOR_WARN}]",
            border_style=COLOR_WARN,
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )


def print_divider() -> None:
    console.rule(f"[{COLOR_DIM}]cycle closed[/{COLOR_DIM}]", style=COLOR_DIM)


def print_cost(input_tokens: int, output_tokens: int) -> None:
    cost_in = (input_tokens / 1_000_000) * 3
    cost_out = (output_tokens / 1_000_000) * 15
    total = cost_in + cost_out
    table = Table.grid(padding=(0, 2))
    table.add_column(style=COLOR_ACCENT)
    table.add_column(style=COLOR_TEXT)
    table.add_row("ingress", f"{input_tokens:,} tokens")
    table.add_row("egress", f"{output_tokens:,} tokens")
    table.add_row("est. consumption", f"${total:.4f}")
    console.print(
        Panel(
            table,
            title=f"[{COLOR_SECONDARY}]Telemetry :: Compute Cost[/{COLOR_SECONDARY}]",
            border_style=COLOR_SECONDARY,
            box=box.ROUNDED,
        )
    )


def print_retry(attempt: int, max_retries: int, reason: str) -> None:
    console.print(
        _safe_text(f"\n  [{COLOR_WARN}]↺ Retry Attempt [{attempt}/{max_retries}] ❯ {reason}[/{COLOR_WARN}]")
    )


def print_info(msg: str) -> None:
    console.print(
        Panel(
            _safe_text(msg),
            title=f"[bold {COLOR_ACCENT}]System Notice[/bold {COLOR_ACCENT}]",
            border_style=COLOR_ACCENT,
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )


def print_warning(msg: str) -> None:
    console.print(
        Panel(
            _safe_text(msg),
            title=f"[bold {COLOR_WARN}]Warning[/bold {COLOR_WARN}]",
            border_style=COLOR_WARN,
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )


def print_goodbye() -> None:
    console.print(
        Panel(
            Text("Session terminated. RANGE runtime state preserved.", style=f"bold {COLOR_ACCENT}"),
            border_style=COLOR_SECONDARY,
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )


def print_interrupted() -> None:
    print_warning("Interrupt signal caught. Send Ctrl+C again to abort runtime.")


# ─── Spinner ──────────────────────────────────────────────

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

_spinner_thread: threading.Thread | None = None
_spinner_stop = threading.Event()


def start_spinner(label: str = "Processing") -> None:
    global _spinner_thread
    if _spinner_thread is not None:
        return
    _spinner_stop.clear()

    def _run() -> None:
        frame = 0
        _safe_stdout_write(f"\n  {SPINNER_FRAMES[0]} {label}...")
        while not _spinner_stop.is_set():
            time.sleep(0.08)
            frame = (frame + 1) % len(SPINNER_FRAMES)
            _safe_stdout_write(f"\r  {SPINNER_FRAMES[frame]} {label}...")

    _spinner_thread = threading.Thread(target=_run, daemon=True)
    _spinner_thread.start()


def stop_spinner() -> None:
    global _spinner_thread
    if _spinner_thread is None:
        return
    _spinner_stop.set()
    _spinner_thread.join(timeout=1)
    _spinner_thread = None
    _safe_stdout_write("\r\033[K")


# ─── Plan approval display ──────────────────────────────────


def print_plan_for_approval(plan_content: str) -> None:
    lines = plan_content.split("\n")
    max_lines = 60
    preview = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        preview += f"\n\n... ({len(lines) - max_lines} more lines)"
    console.print(
        Panel(
            _safe_text(preview),
            title=f"[bold {COLOR_ACCENT}]Proposed Execution Graph[/bold {COLOR_ACCENT}]",
            border_style=COLOR_ACCENT,
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


def print_plan_approval_options() -> None:
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("id", style=f"bold {COLOR_ACCENT}", no_wrap=True)
    table.add_column("action", style=f"bold {COLOR_TEXT}")
    table.add_column("detail", style=COLOR_DIM)
    table.add_row("1", "Flush Context & Execute", "zero-shot restart with autonomous workspace edits")
    table.add_row("2", "Direct Execution", "retain session context, auto-accept code mutations")
    table.add_row("3", "Step-by-Step Approval", "retain context, prompt confirmation per file change")
    table.add_row("4", "Iterate Strategy", "stay in planning loop to incorporate constraints")
    console.print(
        Panel(
            table,
            title=f"[bold {COLOR_WARN}]Select Strategy Route[/bold {COLOR_WARN}]",
            border_style=COLOR_WARN,
            box=box.ROUNDED,
        )
    )


# ─── Sub-agent display ──────────────────────────────────────


def print_sub_agent_start(agent_type: str, description: str) -> None:
    console.print(
        Panel(
            _safe_text(description),
            title=f"[bold {COLOR_SECONDARY}]Sub-agent Spawned :: {agent_type}[/bold {COLOR_SECONDARY}]",
            border_style=COLOR_SECONDARY,
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )


def print_sub_agent_end(agent_type: str, _description: str) -> None:
    console.print(
        Panel(
            "State synchronized to parent runtime.",
            title=f"[bold {COLOR_SUCCESS}]Sub-agent Completed :: {agent_type}[/bold {COLOR_SUCCESS}]",
            border_style=COLOR_SUCCESS,
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )


# ─── multiagent 编排进度 ────────────────────────────────────
# 由 agents/multiagents/trace.py 的 use_host_ui() 挂上去。用 Text 而不是
# f-string + markup：节点文本里有 `[code]`、`[... 监督层净化 ...]` 这种方括号，
# 走 markup 会被 rich 当成标签解析，轻则丢字符重则报错。
MULTIAGENT_NODE_ICON = {
    "plan": "◆",
    "route": "→",
    "worker": "▸",
    "join": "┄",
    "verify": "✓",
    "synthesize": "★",
}

_MULTIAGENT_LABEL_WIDTH = 24


def print_multiagent_node(node: str, text: str) -> None:
    """编排层一个节点的一行主输出。"""
    line = Text()
    line.append(f"{MULTIAGENT_NODE_ICON.get(node, '•')} ", style=f"bold {COLOR_ACCENT}")
    line.append("multiagent ", style=COLOR_SECONDARY)
    line.append(f"{node:<10}", style=f"bold {COLOR_ACCENT}")
    line.append(" ")
    line.append(_safe_text(text), style=COLOR_TEXT)
    console.print(line)


def print_multiagent_detail(text: str) -> None:
    """编排层节点下的细节行。"""
    console.print(Text(" " * _MULTIAGENT_LABEL_WIDTH + _safe_text(text), style=COLOR_DIM))


def print_memory_entries(memories: list[object]) -> None:
    table = Table(box=box.ROUNDED, header_style=f"bold {COLOR_ACCENT}", border_style=COLOR_SECONDARY)
    table.add_column("Type", style=f"bold {COLOR_ACCENT}", no_wrap=True)
    table.add_column("Key", style=f"bold {COLOR_TEXT}")
    table.add_column("Content Representation", style=COLOR_DIM)
    for m in memories:
        table.add_row(
            _safe_text(getattr(m, "type", "")),
            _safe_text(getattr(m, "name", "")),
            _safe_text(getattr(m, "description", "")),
        )
    console.print(
        Panel(
            table,
            title=f"[bold {COLOR_ACCENT}]Episodic & Semantic Memories[/bold {COLOR_ACCENT}]",
            border_style=COLOR_ACCENT,
            box=box.ROUNDED,
        )
    )


def print_skill_entries(skills: list[object]) -> None:
    table = Table(box=box.ROUNDED, header_style=f"bold {COLOR_ACCENT}", border_style=COLOR_SECONDARY)
    table.add_column("Skill Vector", style=f"bold {COLOR_ACCENT}", no_wrap=True)
    table.add_column("Source", style=COLOR_TEXT, no_wrap=True)
    table.add_column("Scope", style=COLOR_SECONDARY, no_wrap=True)
    table.add_column("Docstring / Purpose", style=COLOR_TEXT)
    for s in skills:
        name = getattr(s, "name", "")
        tag = f"/{name}" if getattr(s, "user_invocable", False) else name
        table.add_row(
            _safe_text(tag),
            _safe_text(getattr(s, "source", "")),
            _safe_text(getattr(s, "context", "")),
            _safe_text(getattr(s, "description", "")),
        )
    console.print(
        Panel(
            table,
            title=f"[bold {COLOR_ACCENT}]Skill Graph / Capability Registry[/bold {COLOR_ACCENT}]",
            border_style=COLOR_ACCENT,
            box=box.ROUNDED,
        )
    )


# ─── Tool icons and summaries ───────────────────────────────

_TOOL_ICONS = {
    "read_file": "📄",
    "write_file": "📝",
    "edit_file": "⚡",
    "list_files": "🗂️ ",
    "grep_search": "🔎",
    "run_shell": "🖥️ ",
    "skill": "🔮",
    "skill_create": "🧬",
    "skill_evolve": "📈",
    "agent": "🤖",
}


def _get_tool_icon(name: str) -> str:
    return _TOOL_ICONS.get(name, "⚙️ ")


def _get_tool_summary(name: str, inp: dict) -> str:
    if name == "read_file":
        return inp.get("file_path", "")
    if name == "write_file":
        return inp.get("file_path", "")
    if name == "edit_file":
        return inp.get("file_path", "")
    if name == "list_files":
        return inp.get("pattern", "")
    if name == "grep_search":
        return f'"{inp.get("pattern", "")}" in {inp.get("path", ".")}'
    if name == "run_shell":
        cmd = inp.get("command", "")
        return cmd[:60] + "..." if len(cmd) > 60 else cmd
    if name == "skill":
        return inp.get("skill_name", "")
    if name == "skill_create":
        return inp.get("name", "")
    if name == "skill_evolve":
        return inp.get("skill_name", "")
    if name == "agent":
        return f'[{inp.get("type", "general")}] {inp.get("description", "")}'
    return ""