# TierMem 实现报告

> 位置：`agents/TierMem/`
> 设计来源：*From Lossy to Verified: A Provenance-Aware Tiered Memory for Agents*（arXiv:2602.17913）
> 状态：包与 `agents/agent.py` 接线均已完成；主循环级端到端尚未在项目真实运行环境验证

---

## 一、这份实现要解决什么

原记忆系统有两个结构性缺口：

1. **写入即定稿，没有证据链。** 长期记忆（`agents/memory.py`）每条是一份 Markdown，frontmatter 只有 `name / description / type`。一条记忆无法回答它来自哪次会话、哪段原始对话。检索粒度只有"一行 description"和"整篇 4KB 截断"两档，中间没有可用粒度。
2. **会话折叠是不可逆删除。** `agents/session_memory.py` 配合 `agent.py` 的折叠逻辑，把整段历史压缩成一段 JSON 后**替换掉原文**。原文在任何落盘路径里都不存在，所以"证据不足时升级去读原始日志"这条路没有终点。

TierMem 的做法是把检索当成一次推理时的证据分配：**默认用最便宜的摘要回答，摘要缺少决定性约束时才升级去读不可变的原始日志，并把升级中验证过的结论带溯源写回索引。**

---

## 二、模块清单

合计 1658 行（不含 `agents/agent.py` 与 `agents/main.py` 的改动）。

| 文件 | 行数 | 职责 | 是否含 LLM 调用 |
| --- | --- | --- | --- |
| `config.py` | 209 | 全部常量与可调参数。**全是字面量，不读任何环境变量** | 否 |
| `prompts.py` | 126 | 全部提示词模板，可调部分从 `config` 插值 | 否 |
| `store.py` | 108 | Tier-2 不可变分页原始日志 | 否 |
| `index.py` | 185 | Tier-1 摘要索引、溯源链接、向量检索、`cosine` | 否 |
| `llm.py` | 152 | `LLMProtocol` 协议、`HttpLLM`、`AgentChatLLM`、`extract_json`。**不含任何容灾** | 是 |
| `router.py` | 87 | 充分性路由、评测判官 `judge` | 是 |
| `writeback.py` | 203 | 确定性溯源链接、`tokens_of`、SKIP/UPDATE/ADD 裁决 | 是 |
| `tiermem.py` | 512 | 编排器：`ingest` / `query` / `commit` 与接入辅助 | 是 |
| `__init__.py` | 76 | 公开 API 导出 | 否 |

依赖关系是单向的：`store` 与 `index` 不依赖 `llm`，向量由调用方算好传入。因此这两个模块可以直接单测，不需要任何模型或网络。

---

## 三、数据模型与存储布局

数据目录：`<cwd>/.bear/tier_memory/<session_id>/`，与 `agents/memory.py` 的 `.bear/memory` 并列，互不覆盖。

**按会话隔离。** `config.SESSION_ISOLATION = True` 时，每个会话写进自己的子目录：

```
.bear/tier_memory/
  b9e885b7/                    ← 一个会话的全部记忆，可整目录删除
    tier2_pages.jsonl
    tier1_units.json
    tier2_embeddings.json
  3f2a91cc/                    ← 另一个会话
    ...
```

目录由 `config.session_root(session_id)` 决定，`Agent._tier_memory()`（L955）传入 `self.session_id`。隔离的效果：会话之间不互相召回、页号在各自目录内从 `p000001` 独立开始、删掉一个会话不影响其他会话。

关掉隔离（`SESSION_ISOLATION = False`）或 `session_id` 为空时退化为项目级共享目录 `.bear/tier_memory/`，此时所有会话共用一份索引，可跨会话召回，但无法单独清理。

`--resume` 会接管被恢复会话的 `session_id`（`main.py` L421 附近），否则恢复出来的历史没有对应的记忆目录。

会话存档（`agents/session.py`）同样放在项目内 `.bear/sessions/`，不放用户主目录：沙箱下主目录只授予沙箱组读 + 执行权限（`CodexSandboxUsers:(RX)`），`mkdir` 会抛 `PermissionError: [WinError 5] 拒绝访问`、写文件抛 `[Errno 13] Permission denied`，导致会话存档与 `--resume` 整条链路失效。工具大结果（`agent.py::_persist_large_result`）同理改写到 `.bear/tool-results/`——那一处没有 try 保护，会直接打断工具循环。

| 文件 | 形态 | 写入方式 |
| --- | --- | --- |
| `tier2_pages.jsonl` | 一行一个 JSON 对象 | 只追加（`open("a")`），永不修改、永不删除 |
| `tier1_units.json` | 单个 JSON 对象，`{"units": [...]}` | 整体覆写 |
| `tier2_embeddings.json` | `{page_id: [float]}` | 整体覆写 |

字段：

| 对象 | 字段 |
| --- | --- |
| Tier-2 页 | `page_id`（`p%06d`，只增不改）、`text`、`session`（写入时的会话 id，元数据；目录已按会话隔离）、`created_at` |
| Tier-1 单元 | `unit_id`（`u%06d`）、`text`、`embedding`、`provenance`（页号列表）、`created_at`、`updated_at`、`origin`（`ingest` / `writeback`） |

两条硬约束：

- **不存在没有溯源链接的单元。** `Tier1Index.add` 在 `provenance` 为空时直接抛 `ValueError`。这是"每条摘要都能回指原始证据"在代码层面的落点。
- **页号固定位宽且只追加。** 页号一旦分配就不漂移，摘要可以长期稳定引用它。

写入侧生成的摘要单元的溯源是**直接写死的**（摘要就是这一页生成的），不经过模型。只有写回阶段需要把事实映射回页号，那时才用确定性匹配。

---

## 四、执行链路

### 4.1 包内三个阶段

| 阶段 | 方法 | 做了什么 |
| --- | --- | --- |
| 摄入 | `ingest(messages, session, page_tokens)` | 按 token 切页 → 追加落盘 Tier-2 → 为每个新页算向量并缓存 → 按页生成摘要 → 写入 Tier-1 |
| 查询 | `query(question, k, page_budget)` | 召回摘要 → 路由判定 → `ANSWER` 直接用摘要，或 `ESCALATE` 后先读摘要链接的页再最多 `MAX_ESCALATION_HOPS` 步多跳 → 生成回答 |
| 写回 | `commit(question, trace)` | 抽原子事实 → 按与触发查询的相关度截断 → 确定性映射回页号 → 冲突检索 → SKIP/UPDATE/ADD |

`ingest` 的顺序刻意是"先落 Tier-2，再生成摘要"。如果反过来，摘要生成完但原始页没落盘就退出，溯源链接会指向不存在的证据。

`query` 的返回是一个完整 trace（`route` / `steps` / `mode` / `pages_read` / `answer` / `seconds`），便于审计与评测。`mode` 取值：`empty`（索引为空，短路）、`answer`（仅摘要）、`escalated`（升级过）、`disabled`。

### 4.2 `agents/agent.py` 的接入点

实例创建只有一个入口：`Agent._tier_memory()`（L937，懒加载）。另外 `_build_archive_guidance_section`（L474）只读 `self._tier_mem`，不创建。

**写路径（`ingest`）有三个触发源：**

| 触发源 | 调用点 |
| --- | --- |
| 用户输入 `/compact` | `agent.compact()` L761 ← `main.py` L213 |
| 上下文自动超阈值 | `_check_and_compact()` L854，条件 `last_input_token_count > effective_window * AUTO_COMPACT_THRESHOLD`；调用点在 `_chat_anthropic` 与 `_chat_openai` 各一处 |
| 模型自行调用 `compact_context` 工具 | `_execute_compact_context_tool()` L1239 |

三者都汇入 `_archive_conversation`（L856）→ `_archive_anthropic`（L865）/ `_archive_openai`（L873）→ `_archive_messages`（L885）→ `memory.ingest()`。

**读路径（`query`）每轮用户输入执行一次：**

`_chat_anthropic` L1430 / `_chat_openai` L1731 启动 `start_tier_mem_prefetch()`，与主模型调用并行；循环里任务完成后由 `_inject_tier_mem_result()` 注入（L1472 / L1761）。同一条用户消息只注入一次（`tier_mem_consumed`）。

**写回（`commit`）** 在 `_inject_tier_mem_result` 内部：`trace["mode"] == "escalated"` 且 `ENABLE_WRITEBACK` 时后台 `asyncio.create_task(memory.commit(...))`，任务登记进 `_background_tier_tasks`。

**归档之后历史被替换成什么：** `_build_archive_note()`（L1014）生成一段 `<session-archive>` 说明——归档触发源、本次新增页数与页号区间、新增单元数、当前规模，加上 `render_index()` 的索引快照。**这段是纯字符串拼装，不调用模型**，所以 `compact_context` 现在几乎不产生额外成本（原来的折叠要花一次 6000 token 的 side query）。

**不触发 TierMem 的四种情况：** `is_sub_agent=True`（`enabled=False` 且 prefetch 返回 `None`，避免临时任务的原文污染项目记忆）；索引为空（第一次归档之前，零额外开销）；把 `config.ENABLED` 改为 `False`；查询短于 `MIN_QUERY_CHARS`。

---

## 五、配置参数

全部位于 `config.py`，**全是字面量，不读任何环境变量**。调行为直接改这个文件，读代码时不需要去猜环境里设了什么。

凭据与模型名由 `agents/main.py` 单点解析后注入，本包不重复读环境变量：

| 环节 | 位置 |
| --- | --- |
| 解析 | `main.py::_clean_env`（去空白 + **剥行尾注释**）、`main.py::_resolve_api_config`（api_base / api_key / 是否 OpenAI 兼容）、`VECTOR_MODEL` |
| 注入 | `main.py` 构造 `Agent(..., embedding_model=...)`；`Agent.__init__` 保存 `api_base` / `api_key` / `embedding_model` |
| 使用 | `Agent._build_tier_mem_llm()` 用这三项 + `self.model` 构造 `HttpLLM`，再包成 `AgentChatLLM` 交给 `TierMem` |
| 缺失 | 任一项为空时 `_build_tier_mem_llm()` 直接抛 `RuntimeError`，指明去 `.env` 设 `API` / `APIKEY` / `VECTOR_MODEL` |

`TierMem(root, llm)` 的 `llm` 是**必填**的；不传会抛 `RuntimeError("TierMem 需要显式传入 llm；本包不读环境变量")`。

| 组 | 参数 |
| --- | --- |
| 开关 | `ENABLED`、`ENABLE_ROUTER`、`ENABLE_ESCALATION`、`ENABLE_WRITEBACK`、`ENABLE_KEYWORD_FALLBACK` |
| 存储 | `TIERMEM_DIRNAME`、**`SESSION_ISOLATION`**、`TIER2_PAGES_FILENAME`、`TIER1_INDEX_FILENAME`、`TIER2_VECTORS_FILENAME`、`PAGE_ID_PREFIX/WIDTH`、`UNIT_ID_PREFIX/WIDTH`、`memory_root()`、`session_root()` |
| 分页 | `PAGE_TARGET_TOKENS`(350)、**`MIN_PAGE_TOKENS`(32)**、`TOKENS_PER_CJK_CHAR`、`CHARS_PER_TOKEN`、`PAGE_LINE_TEMPLATE` |
| 摘要 | `SUMMARY_MODE`(retention)、`SUMMARY_MAX_TOKENS`、`SUMMARY_WORD_CAP_LOSSY`(60)、`SUMMARY_WORD_CAP_RETENTION`(90)、`SUMMARY_MIN_CHARS`、`SUMMARY_FALLBACK_CHARS` |
| 检索与预算 | `TIER1_TOP_K`(3)、`TIER2_TOP_K`(3)、`ESCALATION_PAGE_BUDGET`(0＝回落 `TIER1_TOP_K`)、`MAX_ESCALATION_HOPS`(3)、`MAX_EVIDENCE_BYTES`、`MAX_SESSION_BYTES`、`PLAN_EVIDENCE_PREVIEW_CHARS`、`MIN_QUERY_CHARS`(2) |
| 调用上限 | `ROUTER_MAX_TOKENS`、`PLAN_MAX_TOKENS`、`ANSWER_MAX_TOKENS`、`JUDGE_MAX_TOKENS`、`FACT_EXTRACT_MAX_TOKENS`(2500)、`MANAGER_MAX_TOKENS` |
| 路由 | `DEFAULT_ROUTE_ON_ERROR`(ESCALATE)、`ROUTE_ANSWER`/`ROUTE_ESCALATE`/`ROUTE_ACTIONS`、`ESCALATE_ON_EMPTY_INDEX` |
| 写回 | `WRITEBACK_MAX_FACTS`(5)、`WRITEBACK_CONFLICT_TOP_K`(3)、`WB_SKIP`/`WB_UPDATE`/`WB_ADD`、`WRITEBACK_FALLBACK_ACTION`、`WRITEBACK_ADD_ON_MISSING_TARGET` |
| 中文分词 | `CJK_START`/`CJK_END`、`CJK_NGRAM`(2)、`MIN_TOKEN_LEN`、`MIN_KEYWORD_OVERLAP`、`EVIDENCE_QUOTE_CLIP` |
| HTTP 与数值 | `LLM_TIMEOUT_SECONDS`(180)、`EMBEDDING_BATCH_SIZE`(16)、`COSINE_ZERO_EPS`(0.0) |
| 凭据与模型名 | **不在本包内**。由 `agents/main.py` 解析后注入，见上一节 |

`ESCALATION_PAGE_BUDGET` 是成本控制的核心项。它限制升级路径最多读取的**去重后**原始页数；没有它，升级读的页可能比"永远读原文"的基线还多，架构就变成负收益。设为 0 时取 `TIER1_TOP_K`（由 `escalation_page_budget()` 解析）。

`MIN_QUERY_CHARS` 刻意**不用"是否含空白"来判定**。`agents/memory.py` 的 `start_memory_prefetch` 用 `re.search(r"\s", query)` 判"多词"，而中文不用空格分词，导致中文查询几乎永远不触发召回。TierMem 改用字符数下限。

---

## 六、与论文的对应关系

| 论文组件 | 是否迁移 | 实现位置与说明 |
| --- | --- | --- |
| Tier-2 不可变分页原始日志 | 已迁移 | `store.py`，稳定页号 |
| Tier-1 摘要 + 向量 + 溯源链接 | 已迁移 | `index.py` |
| 溯源标识符不由模型生成 | 已迁移 | `writeback.deterministic_link`：归一化子串匹配优先，失败退到关键词重叠；两步都失败返回空，调用方据此 REJECT |
| 运行时充分性路由 | 已迁移（提示词版） | `router.sufficiency_router`，判定项为否定、精确值、归因、范围与顺序 |
| 有界升级 | 已迁移 | `tiermem.query` 的多跳循环 + `MAX_ESCALATION_HOPS` + 页预算 |
| 验证式写回 + 溯源继承 | 已迁移 | `writeback.writeback` + `Tier1Index.update` 合并双方 provenance |
| 路由的 SFT + GRPO 训练 | **未迁移** | 需要约 2000 条标注实例、教师蒸馏与 LoRA 训练流程。当前是零样本提示词路由 |
| 基于 Mem0 的向量库 | **未迁移** | 用本地 JSON + 余弦检索替代，保持项目 `dependencies = []` |
| epoch 式 replay 评估协议 | **未迁移** | 那是论文为对齐 benchmark 做的妥协，在线写回更适合本项目 |

两处需要知道的偏差：

- **Tier-1 单元与 Tier-2 页是 1:1 链接**（按页生成摘要）。原型实验显示，这会削弱"溯源引导升级"的优势：读被召回摘要链接的页会退化为"读 top-k 摘要各自那一页"，排序信号弱于直接对页做嵌入检索。论文报告该机制在升级路径上贡献 +4.2pp，其前提是摘要能跨页合并。
- **路由是零样本的。** 论文实测零样本路由的硬例召回率为 44.7%，SFT+GRPO 为 71.7%，且零样本路由有较强的过升级倾向（保守）。原型实验里升级率达 0.80，虽然不损害正确性但显著推高成本。

论文报告的数字（LoCoMo 0.851、token −54.1%、延迟 −60.7%）依赖其模型与数据分布，不能外推为本项目的预期收益。

---

## 七、降级与失败处理

**客户端层（`llm.py`）不含任何容灾机制**：请求失败直接抛异常，响应结构不对直接 `KeyError`/`IndexError`，凭据缺失直接报错。不重试、不降级、不返回空值、不做结果补齐。这是刻意的取舍——出问题时它会在第一次调用就明确失败，而不是静默地用空数据继续跑。

其余层次的降级都是**记忆语义上的决策**，不是客户端容灾：

| 层 | 机制 |
| --- | --- |
| 摄入 | 摘要生成失败或长度不足 → 退化为截取原文前 `SUMMARY_FALLBACK_CHARS` 字符，保证"有页必有单元"；embed 失败 → 该单元向量记空，检索时跳过；页向量缓存损坏 → 当作空缓存重建 |
| 摄入 | `Tier1Index.add` 硬校验 provenance，为空直接抛错——从结构上禁止"无源摘要" |
| 读取 | 索引为空 → `mode="empty"` 短路，不调路由也不调回答；路由调用抛异常或返回非法 JSON → 统一走 `DEFAULT_ROUTE_ON_ERROR`（默认 ESCALATE），并把回复前 200 字符记进 `reason` |
| 读取 | 回答生成失败 → 返回提示文本而非抛异常；证据超长 → `_clip()` **按字节截断后落回字符边界**，不把多字节字符切坏；规划调用失败 → 跳出多跳循环，用已有证据回答 |
| 读取 | 会话注入字节超 `MAX_SESSION_BYTES` → 强制只走 ANSWER 路径，不再升级。**该计数在归档后由 `TierMem.reset_session_budget()` 归零**（`agents/agent.py::_archive_messages` 调用）：历史被替换时之前注入的证据也随之消失，不归零会让升级在耗尽后永久失效 |
| 写回 | 事实抽取调用失败 → `REJECT` 审计记录；解析不出事实（常见于被 `max_tokens` 截断）→ `NO_FACT_PARSED` 并带回复开头 300 字符；**拿不到溯源一律 REJECT，绝不编造页号** |
| 写回 | 裁决调用失败 → `REJECT` 并保留 provenance；返回非法动作 → 回落 `WRITEBACK_FALLBACK_ACTION`；UPDATE 找不到目标 → 默认退化为 `ADD_FALLBACK` 并在日志标记，便于统计退化率 |
| 存储 | 坏 JSON 行只跳过该行，不让整份日志报废；缺 `page_id` 的行跳过；索引文件缺失 / 损坏 / 顶层类型不对 → 当作空索引 |
| Agent | **归档失败保留原历史，不做破坏性替换**（`_archive_messages` 捕获异常后 `return False`）；注入用 try/except 包住 `task.result()`，失败不影响主对话；后台任务登记进集合防止被垃圾回收；子 agent 完全隔离 |

`FACT_EXTRACT_MAX_TOKENS` 默认 2500 是被实证调出来的：设成 800 时模型返回的 JSON 会被截断，`extract_json` 抛出"括号不配对"，一条事实都解析不出来。

---

## 八、已知缺口

按是否会造成数据损失排序。

| # | 缺口 | 影响 | 位置 |
| --- | --- | --- | --- |
| 1 | 后台 `commit` 任务的异常无人消费，只挂了 `add_done_callback(discard)` | commit 抛错时 asyncio 只打一条 "Task exception was never retrieved"，**记忆写入静默丢失** | `agent.py` `_inject_tier_mem_result` |
| 2 | `Tier1Index` 的 `add` / `update` 是"读内存列表 → 整体覆写文件"，没有并发保护 | 同一轮发生两次升级时，后台 commit 并发执行，后写覆盖先写，**丢单元** | `index.py` `save()` |
| 3 | 落盘非原子（`write_text` 直接覆写） | 进程在写入中途被杀会留下半截 JSON；`_load` 会把它当作空索引，属于**静默丢数据** | `index.py` L117、`tiermem.py` L74 |
| 4 | Tier-2 无容量上限。`MAX_SESSION_BYTES` 只限制注入量，不限制存储 | 长期运行日志单调增长；同时 `store.many()` 每次调用都遍历全部页（O(N)） | `store.py` `many()` |
| 5 | `AgentChatLLM.chat` 委托给 agent 的 side query，**没有单独超时** | side query 卡住会让整个 `query` 任务一直挂着（`HttpLLM` 那条路径有超时） | `agent.py` `_build_tier_mem_llm` |
| 6 | `store.many()` 对不存在的页号静默忽略 | 若 provenance 指向已被外部删除的页，会形成"无源摘要"且不报错；无一致性校验 | `store.py` `many()` |
| 7 | evidence 层没有去重 | 升级时 evidence = 摘要文本 + 页正文，`retention` 模式下二者内容接近，注入重复付费 | `tiermem.query` |

第 1 与第 2 条会实际丢数据，建议优先处理。

**已修复（两处，均经真实模型验证）：**

- **大字页读不全（最严重的一条）。** 原 `chunk_messages` 不切碎单条消息，一条 22KB 的工具结果
  会独占一整页，摘要在 90 词内只能覆盖开头；而升级读取时每段证据又被 `MAX_EVIDENCE_BYTES`(8192)
  截断，**实测 63% 的内容永远取不回来**，模型自己会回答「原始页只完整展示到 line 111，之后被截断」。
  现改为**页大小严格有界**：单条消息超限即切碎（优先按换行，单行超限则按字符硬切），
  续段标注 `(n/m)`；判定用与 `store.est_tokens` 完全一致的累计计数（不能逐行相加，逐行取整会系统性偏小）。
  实测：300 行日志的对话由 3 页变 18 页，`超过上限的页 = 0`，最大页 1402 字节（远低于 8192）；
  极端输入（单行 20 万字符无换行）由 1 页变 286 页，同样 `超过上限 = 0`。
- **注入预算不归零。** `session_bytes` 原本只增不减，耗尽后 `over_budget` 会让升级在整个进程
  生命周期内永久失效（即使中间归档过）。新增 `TierMem.reset_session_budget()`，
  由 `agents/agent.py::_archive_messages` 在历史被替换后调用。实测：手工置到上限后查询
  `mode=answer, budget_exhausted=True`；重置后同一问题 `mode=escalated`。

---

## 九、验证状态

**已验证：**

- `py_compile`：`TierMem` 9 个模块 + `agents/agent.py` 全部通过。
- 纯函数自测：`tokens_of` 对中文正确切出二元组；`deterministic_link` 在引文缺失时能通过关键词重叠命中正确页，对无关事实返回空（拒绝）。
- 配置解耦后的行为确认：`agents/TierMem/config.py` 内 `os.environ` 出现次数为 **0**；不传 `llm` 会抛 `RuntimeError("TierMem 需要显式传入 llm；本包不读环境变量")`；`main.py::_clean_env('ecnu-embedding-small # 1024 dimension')` 返回 `'ecnu-embedding-small'`；`Agent.__init__` 已接受 `embedding_model` 参数。
- 注入式端到端：显式构造 `HttpLLM`（模拟 main.py 注入）后，页向量与单元向量均为 1024 维，查询正常返回。
- 打桩导入：项目 `.venv` 不可用、managed 环境缺 openai / anthropic / rich / tqdm，因此用 `meta_path` 打桩伪造这些依赖后 `import agents.agent` 成功，8 个新方法均存在；并用 `inspect.getsource` 确认 `_archive_messages` 确实调用 `memory.ingest`、`_inject_tier_mem_result` 确实调用 `memory.commit`。
- 真实模型回归（`ecnu-max` + `ecnu-embedding-small`）：
  1. 空索引查询 → `mode=empty`，无模型调用，注入块为空；
  2. 摄入 4 条消息 → 1 页 / 1 单元，`render_index` 与 `build_prompt_section` 正常；
  3. 召回 → 回答正确，注入块 10 行；
  4. `commit` → 空（本次未升级，符合预期）。
- 迁移前在 `.workbuddy/Test/memory/` 做过三方对照实验（summary-only / raw-only / tiered），结论记录在项目 memory 里。

**针对上述两个修复的验证（真实模型）：**

- 页大小有界：正常输入 18 页 / 最大 350 token；极端输入 286 页 / 最大 350 token；两者超限页数均为 0。
- 内容无损：300 行日志逐行核对，缺失 0 行；6 条约束关键词全部命中。
- 深处可读（**绕开路由的隔离验证**，排除路由波动干扰）：把含 `line 290:` 的 `p000017`(1364 字符，
  `_clip` 后完整未被截断) 直接放进证据后回答 → 正确答出 `96ms` 并引用原行。
- 注意：同一次实验里走正常 `query` 路径时，路由判 `ANSWER`（认为摘要足够），模型仅凭摘要硬算
  得出 95ms（正确答案 96ms）。**这是零样本路由的校准问题，不是分页缺陷**——已修复的是「页能不能读到」，
  不是「什么时候该去读」。

**未验证：**

- **主循环级端到端。** 项目 `.venv/Scripts/python.exe` 运行无输出、退出码 1；managed 环境缺运行依赖。需要在项目真实运行环境执行一次 `python -m agents.main`，确认归档（`/compact` 与自动触发）与召回注入的实际表现、以及 token 开销。这是当前最大的未验证项。
- 归档后多轮对话的召回质量（升级率、注入是否干扰主任务）。
- 并发场景（缺口 1、2 的触发条件）。

---

## 十、使用方式

独立运行（自己构造客户端，本包不读环境变量）：

```python
import asyncio
from agents.TierMem import TierMem, HttpLLM

memory = TierMem(
    "./.bear/tier_memory",
    HttpLLM(api_base="https://.../v1", api_key="...",
            chat_model="...", embedding_model="..."),
)
```

接入宿主（复用宿主已有客户端，不重复造）：

```python
from agents.TierMem import TierMem
from agents.TierMem.llm import AgentChatLLM, HttpLLM

async def _chat(system: str, user: str, *, max_tokens: int) -> str:
    return await my_client.chat(system, user, max_tokens=max_tokens)

memory = TierMem(llm=AgentChatLLM(_chat, embed_client))
```

注意 `TierMem` 只接受 `llm` 与 `root`（以及 `summary_mode` / `enabled`），**`llm` 必填**。它不读环境变量、不加载 `.env`、也不解析 `agents/main.py` 的参数——凭据解析全部在 `agents/main.py` 一处完成，然后由 `Agent` 注入。这样只有一个地方读环境变量，不会出现「同一份配置在两处以不同方式解析」的问题。

配置调优直接改 `config.py` 的字面量，例如：

```python
SUMMARY_MODE = "lossy"        # 切到更激进的压缩（实测会丢精确约束）
PAGE_TARGET_TOKENS = 350      # 分页粒度，直接影响摘要是否丢约束
ENABLED = False               # 整体关闭
```

只有凭据与模型名走环境变量：

```bash
APIKEY=...  API=https://.../v1  MODEL=...  VECTOR_MODEL=...
```
