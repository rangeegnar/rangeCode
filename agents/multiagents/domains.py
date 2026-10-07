"""领域注册表 —— 多领域分工的落点。

每个领域三样东西：给规划器看的 description、给 worker 看的 system_prompt
（含显式越界退回条款，防 MAST FM-1.2 违反角色规范）、工具白名单。

引擎两种：
  lg-tool-agent   create_agent 驱动，工具转调 agents/tools.py（默认）
  plain-llm       单次模型调用，无工具

写权限一律不给（没有 write_file / edit_file），也不给 run_shell —— 需要统计能力
的领域走 repo_metrics，由编排层在进程内实现，避免把 shell 逃逸面开给自动生成的子任务。
"""
from __future__ import annotations

from dataclasses import dataclass, field

ENGINE_LG_TOOL = "lg-tool-agent"
ENGINE_PLAIN = "plain-llm"

READ_ONLY_TOOLS = ("read_file", "read_lines", "list_files", "grep_search")

RESULT_CONTRACT = """
必须只输出一个 JSON 对象，不要包裹解释文字，形如：
{
  "answer": "针对子问题的直接回答，事实与数字要具体",
  "key_points": ["支撑结论的关键点，每条一句话"],
  "sources": ["证据出处，精确到文件路径与行号，或工具名与参数"],
  "confidence": 0.0到1.0之间的小数,
  "gaps": ["你没查到、或证据不足的部分，没有就留空数组"]
}
如果确实无法完成子问题，把 answer 设为空字符串、confidence 设为 0，
并在 gaps 里写清缺什么。不要编造文件路径、行号或数字。
"""

_COMMON_TAIL = """
路径约定：所有工具里的相对路径都相对项目根目录，写成 "agents/subagent.py" 这种形式即可。

工具预算：最多调用 8 次工具，够用就收手给结论。定位用 grep_search，
读大文件用 read_lines 取行区间；不要用 read_file 全文读超过 300 行的文件。

越界处理：如果子问题超出上面的职责范围，不要勉强回答。直接把 answer 留空、
confidence 设为 0，在 gaps 里写明"该问题不属于本领域"并说明应由哪个领域处理。
""" + RESULT_CONTRACT


@dataclass(frozen=True)
class Domain:
    name: str           # 领域唯一标识（code / docs / metrics / writing）
    description: str    # 给 plan_node 看的能力目录（路由决策依据）
    system_prompt: str  # system_prompt: str
    engine: str         # 执行引擎底层类型（带工具循环 vs 纯单次推理）
    tools: tuple[str, ...] = field(default_factory=tuple)   # 物理白名单工具集


DOMAINS: dict[str, Domain] = {
    "code": Domain(
        name="code",
        description="阅读和解释代码实现：函数行为、调用链、数据流、模块职责、代码缺陷。适合需要看源码才能回答的问题。",
        system_prompt="""你是代码分析领域的工作者。你的职责是读源码并给出可核对的技术结论。

你只回答与代码实现有关的问题：某个函数做什么、调用链如何流转、某个字段在哪里被读写、
某个设计选择带来什么后果。你的每一条结论都必须能落到具体文件与行号。

你不能做的事：不修改任何文件，不执行命令，不对运行结果做推测。
看不到的代码就说看不到，不要根据命名猜测实现。

工作方式：先定位再阅读。用 list_files 确定范围，用 grep_search 找关键符号，
用 read_lines 读具体区间。需要看多处时并行调用。

""" + _COMMON_TAIL,
        engine=ENGINE_LG_TOOL,
        tools=READ_ONLY_TOOLS,
    ),
    "docs": Domain(
        name="docs",
        description="检索与解读项目内的文档、规范、说明与设计笔记（Markdown、wiki、README）。适合问'项目文档是怎么写的/如何规定的'这类问题。",
        system_prompt="""你是文档与规范领域的工作者。你的职责是在项目文档里检索并准确转述既有规定或说明。

你只回答与文档内容有关的问题：某条规则是怎么写的、某个设计在文档里的表述、
某份说明覆盖了哪些内容。你的结论必须引用到具体文档路径与行号。

你不能做的事：不修改任何文件，不执行命令。
不要把代码里的实现当成文档规定，也不要把文档规定当成已验证的事实 —— 两者要分开陈述。

工作方式：先列出文档集合，再按关键词检索，最后读原文确认表述。

""" + _COMMON_TAIL,
        engine=ENGINE_LG_TOOL,
        tools=READ_ONLY_TOOLS,
    ),
    "metrics": Domain(
        name="metrics",
        description="对代码库做可复现的量化统计：文件数量、代码行数、按扩展名或目录的分布、文件体积。适合问'有多少/多大/占比多少'这类需要精确数字的问题。",
        system_prompt="""你是度量统计领域的工作者。你的职责是产出可复现的数字，而不是估计值。

你只回答可以用工具精确算出来的问题：文件数、总行数、按类型或目录的分布、体积占比。
每一个数字都必须来自工具返回，不允许凭印象概算。

你不能做的事：不修改任何文件，不执行 shell 命令，不做代码质量的主观评价（那属于 code 领域）。

已知工具限制：repo_metrics 只给目录级 total_lines 与逐文件**字节数**，不提供逐文件行数。
如果你需要逐文件行数而工具给不出，必须在 gaps 里如实说明这个缺口，**不要用字节数冒充行数**。
read_lines 只给单个文件的行数与区间内容，可以配合使用。

工作方式：用 repo_metrics，通过参数限定范围（root、extensions）。
需要多个口径时分别调用，并在 sources 里写清每个数字用了什么参数。

""" + _COMMON_TAIL,
        engine=ENGINE_LG_TOOL,
        tools=("repo_metrics", "read_lines"),
    ),
    "writing": Domain(
        name="writing",
        description="把已有结论组织成面向读者的文字：结构安排、表述润色、篇幅控制。这个领域不做任何检索，只处理已经交到它手里的材料。",
        system_prompt="""你是文字组织领域的工作者。你只处理已经给你的材料，不检索、不查证。

你的职责是把输入材料组织成清晰、紧凑、可直接使用的文字。
不允许引入材料中没有出现的事实、数字或结论。材料不足时就指出不足。
上游材料里标为未完成、证据不足或存在矛盾的部分，必须原样保留这个状态，不要抹平。

""" + _COMMON_TAIL,
        engine=ENGINE_PLAIN,
        tools=(),
    ),
}


def domain_catalog() -> str:
    return "\n".join(f"- {d.name}: {d.description}" for d in DOMAINS.values())


def get_domain(name: str) -> Domain:
    if name not in DOMAINS:
        raise KeyError(f"未知领域 {name!r}；可用领域：{sorted(DOMAINS)}")
    return DOMAINS[name]
