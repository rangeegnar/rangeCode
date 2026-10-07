"""TierMem 的全部常量与可调参数。"""

from __future__ import annotations

from pathlib import Path

# ═══════════════════════ 总开关 ═══════════════════════

#: 关掉后 TierMem 不参与任何读写，调用方可以据此短路整条链路。
ENABLED = True

#: 三个阶段的独立开关。
#: 关掉 ROUTER 等价于「永远只用摘要回答」；
#: 关掉 ESCALATION 则永不读原始日志；关掉 WRITEBACK 则索引只停在摄入状态。
ENABLE_ROUTER = True
ENABLE_ESCALATION = True
ENABLE_WRITEBACK = True

#: 溯源链接的子串匹配失败后，是否允许退回到关键词重叠匹配。
#: 关掉可以让「一条事实要么有精确引文来源，要么被拒绝」，适合做严格实验。
ENABLE_KEYWORD_FALLBACK = True


# ═══════════════════════ 存储布局 ═══════════════════════

#: 相对项目根目录的数据目录，与 agents/memory.py 的 ``.range/memory`` 并列，互不覆盖。
TIERMEM_DIRNAME = "tier_memory"

#: Tier-2 追加日志文件名。页只追加、不修改、不删除。
TIER2_PAGES_FILENAME = "tier2_pages.jsonl"

#: Tier-1 索引文件名。
TIER1_INDEX_FILENAME = "tier1_units.json"

#: Tier-2 页向量缓存文件名，避免每次升级都重算 embedding。
TIER2_VECTORS_FILENAME = "tier2_embeddings.json"

#: page_id 的前缀与数字位宽，例如 ``p000001``。位宽定了就不要改，否则历史页号会串。
PAGE_ID_PREFIX = "p"
PAGE_ID_WIDTH = 6

#: unit_id 的前缀与数字位宽，例如 ``u000001``。
UNIT_ID_PREFIX = "u"
UNIT_ID_WIDTH = 6


#: 是否按会话隔离存储。
#:
#: True：每个会话写进 ``.range/tier_memory/<session_id>/``。会话之间互不召回，
#: 可以单独删除某个会话的记忆，页号在各自目录内独立。
#: False：所有会话共用 ``.range/tier_memory/`` 一份索引，跨会话能互相召回，
#: 但无法单独清理某个会话，页号会跨会话连续增长。
SESSION_ISOLATION = True


def memory_root(cwd: Path | str | None = None) -> Path:
    """返回 TierMem 的项目级根目录（``<cwd>/.range/tier_memory``），不存在时创建。"""
    base = Path(cwd) if cwd is not None else Path.cwd()
    root = base / ".range" / TIERMEM_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def session_root(session_id: str | None = None, cwd: Path | str | None = None) -> Path:
    """返回某个会话的存储目录，不存在时创建。

    ``session_id`` 为空或 ``SESSION_ISOLATION`` 为 False 时退化为项目级共享目录。
    """
    root = memory_root(cwd)
    if not SESSION_ISOLATION or not session_id:
        return root
    path = root / session_id
    path.mkdir(parents=True, exist_ok=True)
    return path


# ═══════════════════════ 分页 ═══════════════════════

#: 每页目标 token 数。调小会让 Tier-1 单元更细、摘要丢约束更少，但单元数量线性增长。
#: 实测 900 时整个对话只切出 2 页，摘要会丢掉精确值与否定；350 时能保住。
PAGE_TARGET_TOKENS = 350

#: 页大小的下界，防止传入过小的 page_tokens 把分页切成碎片。
MIN_PAGE_TOKENS = 32

#: token 估算权重（只用于切页，不用于计费）。中文按字计 1，其余字符按 CHARS_PER_TOKEN 计。
TOKENS_PER_CJK_CHAR = 1
CHARS_PER_TOKEN = 4

#: 页格式：``[时间] 说话人: 内容``。
PAGE_LINE_TEMPLATE = "[{ts}] {speaker}: {text}"


# ═══════════════════════ 摘要生成（写入侧） ═══════════════════════

#: "lossy"：只要求抓住主题与方向，允许丢精确值，压缩率最高。
#: "retention"：强制保留精确数值、否定/禁止、归因、顺序约束，压缩率略低但可验证。
SUMMARY_MODE = "retention"
SUMMARY_MODES = ("lossy", "retention")

#: 摘要调用的输出上限与词数上限（按模式分开，便于对照实验）。
SUMMARY_MAX_TOKENS = 260
SUMMARY_WORD_CAP_LOSSY = 60
SUMMARY_WORD_CAP_RETENTION = 90

#: 摘要的最小可用长度；低于它视为生成失败。
SUMMARY_MIN_CHARS = 12

#: 摘要不可用时退化为截取原始页的前 N 个字符，避免出现「有页无单元」的空洞。
SUMMARY_FALLBACK_CHARS = 400


# ═══════════════════════ 检索与成本预算 ═══════════════════════

#: 每次查询从 Tier-1 召回的摘要单元数。它同时是升级路径的证据预算上限。
TIER1_TOP_K = 3

#: 单轮在 Tier-2 上按向量检索的页数上限。
TIER2_TOP_K = 3

#: 升级路径最多读取的**去重后**原始页数。这是成本控制的核心项：
#: 没有它，升级读的页可能比「永远读原始日志」的基线还多，架构就是负收益。
#: 设为 0 时取 TIER1_TOP_K。
ESCALATION_PAGE_BUDGET = 0

#: 溯源引导升级之后，最多再做几轮多跳检索（论文的 T_max）。
MAX_ESCALATION_HOPS = 3

#: 单条证据注入时截断的字节数，防止一条超大页吃掉整个上下文。
MAX_EVIDENCE_BYTES = 8_192

#: 单次会话最多注入的 TierMem 字节数；超限后只允许 ANSWER 路径，不再升级。
MAX_SESSION_BYTES = 60 * 1024

#: 证据拼接时每段在「规划下一步搜什么」的提示里保留的字符数。
PLAN_EVIDENCE_PREVIEW_CHARS = 1_200

#: 查询短于这个字符数就不触发召回。刻意不用「是否含空白」判断：
#: 中文不用空格分词，用空白判定会让中文查询几乎永远不触发（agents/memory.py 的既有缺陷）。
MIN_QUERY_CHARS = 2


def escalation_page_budget() -> int:
    """升级路径的页预算，0 表示跟随检索深度。"""
    return ESCALATION_PAGE_BUDGET if ESCALATION_PAGE_BUDGET > 0 else TIER1_TOP_K


# ═══════════════════════ 模型调用的输出上限 ═══════════════════════

ROUTER_MAX_TOKENS = 300
PLAN_MAX_TOKENS = 200
ANSWER_MAX_TOKENS = 600
JUDGE_MAX_TOKENS = 300

#: 事实抽取的输出上限。设小了会截断 JSON，导致一条事实都解析不出来（实测 800 就会）。
FACT_EXTRACT_MAX_TOKENS = 2_500

#: 写回裁决（Memory Manager）的输出上限。
MANAGER_MAX_TOKENS = 400


# ═══════════════════════ 路由 ═══════════════════════

#: 解析失败或动作非法时的默认动作。ESCALATE 表示「拿不准就去查原文」。
DEFAULT_ROUTE_ON_ERROR = "ESCALATE"

#: 合法动作集合。S=仅用摘要回答，R=升级读原始日志。
ROUTE_ANSWER = "ANSWER"
ROUTE_ESCALATE = "ESCALATE"
ROUTE_ACTIONS = (ROUTE_ANSWER, ROUTE_ESCALATE)

#: 索引为空时是否直接升级。
ESCALATE_ON_EMPTY_INDEX = True


# ═══════════════════════ 写回 ═══════════════════════

#: 一次升级最多沉淀几条事实。抽取出的原子事实通常远多于这个数（实测约 39 条），
#: 这里按「与触发查询的相关度」排序后截断，只保留真正被这次问题需要的那几条。
WRITEBACK_MAX_FACTS = 5

#: 冲突检索时召回多少个相似单元作为裁决上下文。
WRITEBACK_CONFLICT_TOP_K = 3

#: 三种写回动作。
WB_SKIP = "SKIP"
WB_UPDATE = "UPDATE"
WB_ADD = "ADD"
WB_ACTIONS = (WB_SKIP, WB_UPDATE, WB_ADD)

#: 裁决结果不可用时的兜底动作。
WRITEBACK_FALLBACK_ACTION = WB_ADD

#: UPDATE 找不到目标单元时，是退化成 ADD 还是直接丢弃。
WRITEBACK_ADD_ON_MISSING_TARGET = True


# ═══════════════════════ 溯源链接（中文分词） ═══════════════════════

#: 中文区间，用于判断 token 是否含 CJK。
CJK_START = 0x4E00
CJK_END = 0x9FFF

#: CJK 连续片段切成的 n-gram 长度。
#: 注意：Python 的 ``\w`` 已包含 CJK，直接 split 不会把中文切开，
#: 因此必须自己补 n-gram，否则「引文被改写」的降级匹配对中文永远命中不了。
CJK_NGRAM = 2

#: 非 CJK token 的最短长度。
MIN_TOKEN_LEN = 2

#: 关键词重叠命中的最低重叠数。
MIN_KEYWORD_OVERLAP = 1

#: 引文片段在做子串匹配前保留的最大字符数。
EVIDENCE_QUOTE_CLIP = 200


# ═══════════════════════ HTTP 与数值 ═══════════════════════

#: 单次请求超时。
LLM_TIMEOUT_SECONDS = 180.0

#: 向量接口的批量大小。
EMBEDDING_BATCH_SIZE = 16

#: 余弦相似度的零向量保护阈值：任一向量模长为 0 时按 0 处理，避免除零。
COSINE_ZERO_EPS = 0.0
