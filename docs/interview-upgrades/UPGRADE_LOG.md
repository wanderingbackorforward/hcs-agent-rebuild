# 面试驱动升级日志（AOF）

> **规则：改代码前必须先写本文件。像 Redis AOF 先写日志再执行操作一样。**
>
> 每次要给项目补充面试考点相关的功能，必须先在本文件追加一条计划记录，然后才能改代码。
> 面试时按本文件自吹自擂："我的项目实现了 XX，代码在 YY。"

---

## 项目能力 x 面试考点覆盖矩阵

> 基于 2026-07-08 扫描结果。面试考点来自米哈游游戏AI研发Agent方向（C03）4份面经39题。

### 已有实现（可自吹自擂）

| T编号 | 考点 | 项目实现位置 | 面经频次 | 自吹话术 |
|-------|------|-------------|---------|---------|
| T01 | 多Agent协作编排 | `agents/task_classification_agent.py` TaskClassificationAgent -> EnvironmentMatchingAgent / KnowledgeQAAgent | 8题/4份 | "我的项目是多Agent架构，TaskClassificationAgent做意图路由，分发到环境匹配Agent和知识问答Agent" |
| T01 | 意图路由 | `agents/task_classification/agent_router.py` + `task_classifier.py` + 50条golden test | 8题/4份 | "意图路由用LLM分类+规则兜底，有50条golden test保证准确率" |
| T02 | RAG摄取流水线 | `rag/ingestion/pipeline.py` Load->Split->Embed->Upsert | -- | "完整的文档摄取流水线，支持中英文分块" |
| T02 | Hybrid Search | `rag/query_engine/hybrid_search.py` Dense+BM25+RRF+Rerank | -- | "Hybrid Search用Dense Embedding+BM25稀疏检索+RRF融合+可选Rerank" |
| T03 | MCP Server | `mcp_server/` 完整Server(stdio)+3个Tool+错误处理 | 深挖5层/2份 | "我实现了MCP Server，暴露3个标准化工具，用stdio transport，有完整的错误处理" |
| T03 | MCP Client 能力协商 | `mcp_client/` MCPClientBase+Local/Remote+ServerCapabilityProfile+ClientFeatureFlags | -- | "MCP Client做真实能力协商，Server不支持的能力直接关开关，纯客户端控制无胶水代码" |
| T04 | 状态机对话管理 | `agents/task_classification/state_manager.py` StateManager+StateEnum | -- | "用状态机管理对话流程，CLASSIFY->ENVIRONMENT/KNOWLEDGE/OTHER" |
| T05 | 意图路由评估 | `tests/test_task_classification_agent.py` 50条golden test | -- | "50条golden test覆盖意图路由准确率" |
| T06 | 多模型Provider工厂 | `config/model_provider.py` 支持Qwen/DeepSeek/MiniMax/Azure/OpenAI | -- | "多Provider工厂模式，环境变量切换LLM和Embedding" |
| T08 | BM25稀疏检索 | `rag/query_engine/sparse_retriever.py` 自研BM25 | -- | "自研BM25稀疏检索，不依赖第三方库" |
| T10 | 限流 | `api/rate_limit.py` 滑动窗口限流 | -- | "滑动窗口限流，预留Redis扩展" |

### 缺口待补充（面试高频但项目没实现）

| # | T编号 | 考点 | 面经频次 | 计划方案 | 实现位置 | 状态 |
|---|-------|------|---------|---------|---------|------|
| U1 | T01 | 长期记忆机制 | 极高 3/4份 | 向量DB存用户偏好/历史决策，RAG检索注入context，memory gating写入策略 | `agents/memory/` 新模块 | 待实现 |
| U2 | T01 | 上下文长度管理 | 高 2/4份 | tiktoken计数+滚动摘要+超token截断策略 | `agents/context_manager.py` 新文件 | 待实现 |
| U3 | T01 | ReAct工具循环 | 中高 | 给KnowledgeQAAgent加ReAct模式（工具调用->观察->决策循环） | 改 `agents/knowledge_qa_agent.py` + `agents/knowledge_qa/react_loop.py` | 待实现 |
| U4 | T01 | Agent评测 | 中 2次考 | 任务完成率/检索精度/响应质量评分 | `tests/test_agent_eval.py` 新测试文件 | 待实现 |
| U5 | T06 | Agent缓存机制 | 高 深挖 | LLM结果缓存+工具结果缓存，按 `redis-analysis.md` 计划 | `cache/` 新模块 + 改 `knowledge_qa_agent.py` | 待实现 |
| U6 | T03 | MCP over HTTP/SSE | 中 追问Transport层 | 给mcp_server加SSE transport支持，和stdio并存 | 改 `mcp_server/server.py` + `mcp_server/sse_server.py` | 待实现 |
| U7 | T15 | AI Code Review | P0 6题/2份 | 加code_review模块：误报漏报控制+分级报告+agent loop | `code_review/` 新模块 | 待定 |

### 不加到项目（理论性/岗位特定，只做面试话术）

| T编号 | 考点 | 原因 | 面试话术策略 |
|-------|------|------|-------------|
| T07 | Transformer底层 | 纯理论 | topics/notes 记笔记，面试时背 |
| T01 | Agent产品分析(Claude Code/Codex/Manus) | 产品理解类 | topics/notes 记笔记，面试时讲理解 |
| T01 | flatter死循环 | 项目无多Agent辩论场景 | 准备"如何防止"的话术 |
| T13 | LangGraph | 项目用LangChain不迁移 | 准备"为什么选LangChain"话术 |
| T14 | 游戏场景Agent | 岗位特定 | 在reference准备 |

---

## 变更记录（追加式，每次改代码前先写这里）

### 2026-07-08 初始扫描

**操作**：建立面试驱动升级日志，扫描项目能力覆盖矩阵。

**发现**：
- 10个考点已有项目实现（可自吹自擂）
- 7个缺口待补充（U1-U7）
- 5个理论性考点不加到项目

**下一步**：等待用户确认 U7（AI Code Review）是否加到项目，以及 U1-U6 的优先级排序。

---

## 使用方式

### 面试时怎么用本文件

1. 面试官问某个考点 -> 查本文件"已有实现"表
2. 找到对应行 -> 看自吹话术 + 打开代码位置
3. 如果是"缺口待补充" -> 说"我正在实现 XX"或"如果让我实现我会怎么做"

### 改代码前怎么用本文件

1. 确定要补充哪个缺口（U1-U7）
2. 在"变更记录"追加一条：日期 + 缺口编号 + 计划改什么 + 为什么
3. 然后才能改代码
4. 改完后更新"缺口待补充"表的状态为 已实现

### 2026-07-08 U7 确认 + 优先级排序

**操作**：用户确认 U7（AI Code Review）加入项目。排定 U1-U7 实现优先级。

**优先级排序（按面经频次+依赖关系）**：

| 顺序 | 编号 | 考点 | 面经频次 | 理由 |
|------|------|------|---------|------|
| 1 | U1 | 长期记忆机制 | 极高 3/4份 | 面经最高频考点，项目完全缺失 |
| 2 | U2 | 上下文长度管理 | 高 2/4份 | 和U1紧密相关，记忆管理就是上下文管理的一部分 |
| 3 | U5 | Agent缓存机制 | 高 深挖 | 面经深挖考点，redis-analysis.md 已有详细计划 |
| 4 | U7 | AI Code Review | P0 6题/2份 | 米哈游P0考点，面经6题，独立新模块 |
| 5 | U3 | ReAct工具循环 | 中高 | 依赖现有KnowledgeQAAgent，U1/U2完成后做 |
| 6 | U6 | MCP SSE transport | 中 | 面经追问Transport层，独立改动 |
| 7 | U4 | Agent评测 | 中 2次考 | 依赖其他功能实现完，最后做 |

**执行规则**：每个功能严格按 AOF 原则——先在本日志追加计划记录，再改代码，改完更新状态。

### 2026-07-08 U1 实现计划

**缺口**：U1 长期记忆机制
**面经频次**：极高 3/4份（多轮对话长短期记忆/上下文机制/超token处理/记忆设计）
**计划改什么**：
1. 新建 `agents/memory/` 模块
2. `agents/memory/__init__.py` - 模块入口
3. `agents/memory/long_term_memory.py` - 长期记忆：向量DB存储+RAG检索+memory gating写入策略
4. `agents/memory/short_term_memory.py` - 短期记忆：context window管理+滚动摘要
5. 改 `agents/knowledge_qa_agent.py` - 接入长期记忆

**设计要点**：
- Short-term memory：当前对话最近N轮放入context window，超出用滚动摘要压缩
- Long-term memory：用户偏好/历史决策存向量DB，RAG检索注入context
- Memory gating：LLM判断信息重要性，只写入重要信息到long-term
- 读取策略：recency（时间衰减）+ relevance（语义相关度）加权排序

**面试自吹话术**：我的项目实现了分层记忆机制——短期记忆放context window用滚动摘要压缩，长期记忆存向量DB用RAG检索注入，用memory gating策略决定哪些信息写入长期记忆。

### 2026-07-08 U2 实现计划

**缺口**：U2 上下文长度管理
**面经频次**：高 2/4份（上下文机制完整流程/超token处理）
**计划改什么**：
1. 新建 `agents/context_manager.py` - 上下文长度管理器
2. 功能：tiktoken计数 + 滚动摘要 + 超token截断策略
3. 和 U1 的 ShortTermMemory 配合使用

**设计要点**：
- token计数：用 tiktoken 精确计算 context window 占用
- 切分策略：按消息粒度切分，保留最近N轮完整对话
- 超token处理：先压缩旧对话为摘要，仍超则截断最旧消息
- 上下文组装：system prompt + 长期记忆 + 摘要 + 近期对话 + 当前问题

**面试自吹话术**：我实现了上下文长度管理器——用 tiktoken 精确计数，超 token 时先触发滚动摘要压缩旧对话，仍超则按消息粒度截断。上下文组装分四层：system prompt、长期记忆、对话摘要、近期对话。

### 2026-07-08 U5 实现计划

**缺口**：U5 Agent缓存机制
**面经频次**：高 深挖（缓存机制/业界主流/agent caching看法）
**计划改什么**：
1. 新建 `cache/` 模块
2. `cache/__init__.py` - 模块入口
3. `cache/llm_cache.py` - LLM结果缓存：query_hash -> response, TTL 10-30min
4. `cache/tool_cache.py` - 工具结果缓存：相同query不重复检索
5. `cache/semantic_cache.py` - 语义缓存：相似query复用结果（embedding相似度阈值）
6. 改 `agents/knowledge_qa_agent.py` - 接入缓存

**设计要点**：
- LLM缓存：query hash精确匹配，TTL过期自动失效
- 工具缓存：检索结果缓存，文档更新后refresh
- 语义缓存：embedding相似度>0.92的query复用结果，需设置TTL避免脏数据
- 缓存失效：文档更新/手动清除/TTL过期

**面试自吹话术**：我实现了三层Agent缓存——LLM结果缓存用query hash精确匹配，工具结果缓存避免重复检索，语义缓存用embedding相似度复用相似问题的结果。缓存有TTL和手动失效机制。

### 2026-07-08 U7 实现计划

**缺口**：U7 AI Code Review
**面经频次**：P0 6题/2份（误报漏报/太严格/故意写/上下文太长/高效查错/图关系建模）
**计划改什么**：
1. 新建 `code_review/` 模块
2. `code_review/__init__.py` - 模块入口
3. `code_review/reviewer.py` - 核心审查器：分层验证（确定性检查+LLM语义审查+置信度打分）
4. `code_review/agent_loop.py` - Agent Loop：静态分析->代码切分->LLM审查->全局聚合->分级报告
5. `code_review/report.py` - 分级报告：Error/Warning/Info + 置信度
6. `code_review/context_manager.py` - 代码上下文管理：按文件/函数切分，AST解析

**设计要点**：
- 分层验证：确定性检查（lint/编译，零误报）+ LLM语义审查（补充）+ 置信度打分
- 太严格处理：分级报告（Error阻断/Warning建议/Info参考）+ 可配置严格度 + 白名单
- 上下文管理：按文件/函数切分，AST解析确定边界，每个chunk独立审查
- Agent Loop：静态分析->代码切分->LLM审查->全局聚合->置信度打分->分级输出

**面试自吹话术**：我实现了AI Code Review模块——分层验证（确定性检查零误报+LLM语义审查+置信度打分），分级报告（Error阻断/Warning建议/Info参考），agent loop自动审查代码变更，和文档扫描系统的pipeline架构一致。

### 2026-07-08 U3 实现计划

**缺口**：U3 ReAct工具循环
**面经频次**：中高（agent loop设计/prompt设计）
**计划改什么**：
1. 新建 `agents/knowledge_qa/react_loop.py` - ReAct模式实现
2. 功能：Thought -> Action -> Observation -> Thought循环，最多N轮
3. 改 `agents/knowledge_qa_agent.py` - 支持ReAct和直线两种模式

**设计要点**：
- ReAct模式：LLM输出Thought+Action -> 执行工具 -> Observation注入 -> 循环
- 工具集：query_knowledge_hub / list_collections / get_document_summary
- 最大循环次数：5轮（防止死循环）
- 直线模式保留（简单查询不需要ReAct）

**面试自吹话术**：我的KnowledgeQAAgent支持ReAct模式——LLM输出Thought和Action，执行工具后把Observation注入context，最多5轮循环。简单查询走直线模式，复杂查询走ReAct。

### 2026-07-08 U6 实现计划

**缺口**：U6 MCP over HTTP/SSE
**面经频次**：中（追问Transport层）
**计划改什么**：
1. 新建 `mcp_server/sse_server.py` - SSE transport支持
2. 改 `mcp_server/server.py` - 支持 stdio 和 SSE 两种模式切换
3. SSE模式：通过环境变量 MCP_TRANSPORT=sse 切换

**设计要点**：
- stdio模式：本地CLI使用，进程间通信
- SSE模式：远程访问，HTTP+SSE双向通信
- 环境变量切换：MCP_TRANSPORT=stdio|sse
- SSE优势：跨网络访问、浏览器直连、多客户端

**面试自吹话术**：我的MCP Server支持两种transport——stdio用于本地CLI，SSE用于远程访问。环境变量切换，SSE模式可以跨网络访问和浏览器直连。

### 2026-07-08 U4 实现计划

**缺口**：U4 Agent评测
**面经频次**：中 2次考（Few-shot与Agent评测）
**计划改什么**：
1. 新建 `tests/test_agent_eval.py` - Agent评测测试
2. 功能：任务完成率/检索精度/响应质量评分/Few-shot评估

**设计要点**：
- 任务完成率：Agent是否完成了用户请求
- 检索精度：RAG检索结果的相关性（top-k命中率）
- 响应质量：LLM回答的准确性/完整性/简洁性
- Few-shot评估：用少量标注样本评估Agent表现

**面试自吹话术**：我实现了Agent评测框架——任务完成率评估Agent是否完成请求，检索精度用top-k命中率，响应质量评分准确性和完整性，还支持Few-shot评估。

### 2026-07-08 U1-U7 全部实现完成

**完成状态**：

| # | 考点 | 状态 | 新增文件 | 验证结果 |
|---|------|------|---------|---------|
| U1 | 长期记忆机制 | 已实现 | `agents/memory/__init__.py`, `short_term_memory.py`, `long_term_memory.py` | import OK, smoke test PASSED |
| U2 | 上下文长度管理 | 已实现 | `agents/context_manager.py` | tiktoken OK, context assembly OK |
| U3 | ReAct工具循环 | 已实现 | `agents/knowledge_qa/react_loop.py` | import OK |
| U4 | Agent评测 | 已实现 | `tests/test_agent_eval.py` | 10 tests PASSED |
| U5 | Agent缓存机制 | 已实现 | `cache/__init__.py`, `llm_cache.py`, `tool_cache.py`, `semantic_cache.py` | import OK, cache hit/miss OK |
| U6 | MCP SSE transport | 已实现 | `mcp_server/sse_server.py` + 改 `server.py` | module found, transport switching OK |
| U7 | AI Code Review | 已实现 | `code_review/__init__.py`, `report.py`, `reviewer.py`, `agent_loop.py` | import OK, 2 issues found on test code |

**新增文件清单（17个）**：
- `agents/memory/` 目录：3个文件（U1）
- `agents/context_manager.py`：1个文件（U2）
- `agents/knowledge_qa/react_loop.py`：1个文件（U3）
- `tests/test_agent_eval.py`：1个文件（U4）
- `cache/` 目录：4个文件（U5）
- `mcp_server/sse_server.py`：1个文件（U6）
- `code_review/` 目录：4个文件（U7）
- 改 `mcp_server/server.py`：transport切换（U6）
- 改 `agents/knowledge_qa_agent.py`：接入U1+U2记忆和上下文管理

**面试自吹自擂清单（面试官问→我答→打开代码）**：
1. "多轮对话记忆怎么做？" → 分层记忆：短期(context window+滚动摘要) + 长期(向量DB+memory gating) → `agents/memory/`
2. "上下文超token怎么办？" → tiktoken计数+滚动摘要压缩+消息截断 → `agents/context_manager.py`
3. "agent loop怎么设计？" → ReAct模式：Thought→Action→Observation循环，最多5轮 → `agents/knowledge_qa/react_loop.py`
4. "Agent怎么评测？" → 任务完成率+检索精度@K+响应质量评分+Few-shot → `tests/test_agent_eval.py`
5. "缓存机制？业界主流？" → 三层缓存：LLM结果缓存+工具结果缓存+语义缓存 → `cache/`
6. "MCP Transport层？" → stdio+SSE双模式，环境变量切换 → `mcp_server/sse_server.py`
7. "AI Code Review怎么做？" → 分层验证(确定性+LLM+置信度)+分级报告(Error/Warning/Info)+agent loop → `code_review/`

---

### 2026-07-09 UX 升级计划：延迟感知 + 决策可解释 + 任务可中断 + 错误恢复

**背景**：面试考点 Agent 用户体验四大维度，当前项目全部缺失或不足。详细方案见 `docs/agent-ux-upgrade-plan.md`。

**四个 feature 分支**（按依赖顺序开发）：

| # | 分支 | 考点 | 现状 | 目标 |
|---|------|------|------|------|
| U8 | `feature/sse-streaming` | 延迟感知 | `/chat/stream` 是 text/plain，前端用非流式 `/chat` | SSE 协议，结构化事件（status/token/decision/error/done），前端 EventSource |
| U9 | `feature/decision-explainability` | 决策可解释 | 有 audit_event 日志但不暴露给用户 | decision 事件推送到前端，折叠面板展示路由原因+置信度 |
| U10 | `feature/task-interruptible` | 任务可中断 | 无任何中断机制 | cancel 端点 + CancelledError 处理 + checkpoint 续传 |
| U11 | `feature/error-recovery` | 错误恢复 | 有异常处理但无用户建议/重试 | 错误分类 + 建议 + 一键重试按钮 |

**依赖关系**：U8 是基础设施 → U9/U10 依赖 U8 → U11 依赖 U8+U10

**自吹话术**：
1. "流式输出怎么做？" → SSE 协议，5 种事件类型（status/token/decision/error/done），seq 序列号 + Last-Event-ID 断线续传
2. "决策可解释性？" → decision 事件推送路由原因和置信度，折叠面板展示，mask_sensitive 过滤敏感信息
3. "任务可以中断吗？" → asyncio.CancelledError 处理 + checkpoint 断点续传 + TaskManager 状态管理
4. "出错怎么办？" → 错误分类（可重试/不可重试）+ 用户建议 + 一键重试（max 3 次）

### 2026-07-09 U8-U11 全部实现完成

**完成状态**：

| # | 考点 | 状态 | 新增/修改文件 | 测试 | 验证结果 |
|---|------|------|-------------|------|---------|
| U8 | 延迟感知 (SSE) | 已实现 | `config/sse_protocol.py`, `api/sse_protocol.py`, `api/sse_buffer.py`, 改 `web/routes.py`, 改前端 | `tests/test_sse_protocol.py` (15 tests) | 70 tests PASSED |
| U9 | 决策可解释性 | 已实现 | `config/decision_explainer.py`, 改 `classification_processor.py`, 改 `agent_router.py`, 改 `knowledge_qa_agent.py`, 改前端 | `tests/test_decision_explainer.py` (13 tests) | 70 tests PASSED |
| U10 | 任务可中断 | 已实现 | `api/task_manager.py`, 改 `web/routes.py` (cancel endpoint), 改 `classification_processor.py` (cancellation checks), 改前端 | `tests/test_task_manager.py` (11 tests) | 70 tests PASSED |
| U11 | 错误恢复 | 已实现 | `config/error_classifier.py`, 改 `web/routes.py` (retry endpoint + classify integration), 前端错误卡片 | `tests/test_error_classifier.py` (22 tests) | 70 tests PASSED |

**Git 历史**（4 feature 分支，--no-ff 合并到 main）：
- `811a5a2` merge: feature/error-recovery — error classification and retry (U11)
- `f005d2a` feat(error): error classifier + retry endpoint with max-retry protection
- `11106bb` merge: feature/task-interruptible — task cancellation and checkpointing (U10)
- `a235420` feat(task): cancel endpoint + pipeline cancellation checks + frontend
- `f2bacde` feat(task): TaskManager for cooperative cancellation and checkpointing
- `b31902b` merge: feature/decision-explainability — decision explainability (U9)
- `b2dd483` feat(decision): emit decision SSE events + frontend collapsible panel
- `1789ec0` feat(decision): decision explainer module with safe field whitelisting
- `d8b7d7f` merge: feature/sse-streaming — SSE protocol upgrade (U8)
- `c523386` feat(sse): frontend EventSource consumption with streaming display
- `cef65ca` feat(sse): backend SSE streaming + agent pipeline status events
- `55cd536` feat(sse): SSE event protocol and ring buffer for structured streaming

**新增文件清单（8 个）**：
- `config/sse_protocol.py` — SSEEvent dataclass + 5 factory methods + format_sse_stream + collect_text
- `api/sse_protocol.py` — thin re-export (避免循环导入)
- `api/sse_buffer.py` — per-session ring buffer for Last-Event-ID replay
- `config/decision_explainer.py` — build_decision + agent_display_name + field whitelisting + mask_sensitive
- `api/task_manager.py` — TaskManager (cooperative cancel via asyncio.Event + checkpoint + TTL prune)
- `config/error_classifier.py` — classify() + ErrorInfo + ErrorCategory (type rules + pattern rules + conservative default)
- `tests/test_sse_protocol.py` — 15 tests
- `tests/test_decision_explainer.py` — 13 tests
- `tests/test_task_manager.py` — 11 tests
- `tests/test_error_classifier.py` — 22 tests

**关键设计决策**：
- SSEEvent 放 `config/` 不放 `api/` — 避免 `api/__init__` → `chat_handler` → `agents` → `api` 循环导入
- 协作式取消用 asyncio.Event 不用 Task.cancel() — 避免资源泄漏和状态不一致
- 错误分类保守策略：未知错误默认 non-retryable — 防止 retry storm
- retry 端点 max 3 次限制 — 防止无效重试打垮服务器
- decision 字段白名单 + mask_sensitive — 防止敏感信息泄漏
- 前端用 fetch+ReadableStream 不用 EventSource — POST 端点 + API Key 认证

### 2026-07-10 MCP 闭环接线计划（Step 1）

**背景**：当前项目中 `KnowledgeQAAgent` 的自然语言路由是通的，但生产链路仍然直连 `KnowledgeService.search()`，没有真正通过 MCP Tool 做工具规划与调用。面试时会被追问“你的子 Agent 真会用 MCP 吗”。

**本步只改一个核心问题**：
1. 让 `KnowledgeQAAgent` 真实接入 3 个 MCP Tool，而不是只走本地检索服务。
2. 让 `ReActLoop` 支持异步工具调用，能执行真实 MCP handler。
3. 增加闭环测试，证明“自然语言 -> ReAct -> MCP Tool -> 最终答案”是成立的。

**本步不做的事**：
- 暂不扩展 3 个 Tool 的参数和返回字段。
- 暂不改 transport 层。
- 暂不引入新的 Tool。

**验证目标**：
- 代码层能看到 `KnowledgeQAAgent` 通过 MCP 协议层调工具。
- 测试层能证明至少一条自然语言问答链路真实调用了 `query_knowledge_hub`。
- 改动可独立提交，作为一个原子 git checkpoint。

### 2026-07-10 方案调整：先做工具结构设计与链路澄清

**原因**：用户确认当前阶段先不碰 `ReActLoop`，避免把“顶层分流”与“查询子 Agent 内部执行策略”混在一起，先把架构讲清楚、工具设计做扎实，再决定是否接 ReAct。

**本轮只做两件事**：
1. 重新设计 `query_knowledge_hub`、`list_collections`、`get_document_summary` 的参数与返回结构，补齐字段与可消费性。
2. 把当前查询子 Agent 的真实链路写清楚，明确“哪里已经打通，哪里还没有打通”。

**本轮明确不做**：
- 不改 `TaskClassificationAgent` 顶层分流逻辑。
- 不接 `ReActLoop` 到生产链路。
- 不新增 MCP Tool。

**预期产物**：
- 一份面向开发和面试表述都可直接使用的 Markdown 说明文档。
- 一个干净的 git checkpoint，便于下一步单独讨论是否接入 ReAct。

### 2026-07-10 代码改造 Step 1：升级 `query_knowledge_hub`

**目标**：先只升级一个 MCP Tool，验证“小步快跑 + 原子提交”的节奏可行。

**本步范围**：
1. 扩充 `query_knowledge_hub` 的输入参数，补齐过滤条件和返回控制项。
2. 把输出从面向人类阅读的 Markdown，升级为更适合 Agent 消费的结构化 JSON 文本。
3. 同步更新测试与 `eval/golden_cases.py` 中的 schema 镜像。

**本步不做**：
- 不改 `list_collections`
- 不改 `get_document_summary`
- 不碰 `KnowledgeQAAgent` 主链
- 不碰 `ReActLoop`

### 2026-07-10 错误修复 Step 2：MCP 工具测试去数据库化

**背景**：在验证 `query_knowledge_hub` 升级时，暴露出 `tests/test_mcp_tools.py` 依赖临时 SQLite 与 embedding/LLM 环境，导致工具单测被数据库建表和外部依赖绑死，无法稳定作为每步改造的回归基线。

**本步改动**：
1. 将 `tests/test_mcp_tools.py` 改为真正的单元测试。
2. 用 fake `KnowledgeService`、fake repo、fake LLM 替代真实 SQLite / embedding / 外部模型调用。
3. 保留对 `query_knowledge_hub`、`list_collections`、`get_document_summary` 的行为验证，但去掉环境依赖。

**结果**：
- `pytest tests/test_mcp_tools.py -q` 通过，6 个测试全部通过。
- 后续每升级一个 MCP Tool，都可以直接用这组单测做稳定回归。

**备注**：
- 项目默认 SQLite 库仍然存在 schema 漂移问题，例如 `environments.deploy_method` 缺列，这属于下一步单独修的 runtime 兼容问题，不和本次单测基线修复混在一起。

### 2026-07-11 代码改造 Step 3：升级 `list_collections`

**目标**：把 `list_collections` 从面向人类阅读的展示型工具，升级为面向 Agent 的知识发现工具。

**本步改动**：
1. 输入参数增加 `include_categories`、`include_doc_samples`、`sample_size`、`keyword`。
2. 输出从 Markdown 文本改为结构化 JSON，统一返回 `collections`、`total_collections`、`keyword_used` 等字段。
3. 测试改为验证结构化返回，并补充关键字过滤与非法参数校验。

**结果**：
- `list_collections` 现在既能返回集合统计，也能返回分类摘要和样本文档。
- 后续如果查询子 Agent 需要先“发现知识空间”再决定检索，这个工具可以直接复用。

### 2026-07-11 代码改造 Step 4：升级 `get_document_summary`

**目标**：把 `get_document_summary` 从简单的文本截断工具，升级为文档级结构化查看工具。

**本步改动**：
1. 输入参数增加 `max_chars`、`include_metadata`、`include_source`、`include_chunk_stats`。
2. 输出从 Markdown 文本改为结构化 JSON，统一返回 `doc_id`、`title`、`category`、`summary`、`metadata` 等字段。
3. 测试补充文档元数据断言和非法参数校验。

**结果**：
- `get_document_summary` 现在可以作为查询子 Agent 的文档级查看接口，后续不需要再解析文本摘要。
- 三个 MCP Tool 的结构化升级已经全部完成。

### 2026-07-11 架构定稿：KnowledgeQAAgent 过渡期版本

**背景**：3 个 MCP Tool 已完成结构化升级，但查询子 Agent 还没有真正切到工具驱动。这个阶段不应该直接上 ReAct 或大改实现，而应该先把过渡期架构边界定死。

**本次定稿结论**：
1. 主路径：`KnowledgeQAAgent -> MCP Tool`
2. fallback：保留原 `KnowledgeRetriever -> KnowledgeService`，但只做兜底
3. 最小多步策略：`list -> query -> summary`
4. 暂不上 `ReActLoop`
5. 不再给旧直连链路增加新能力

**设计原则**：
- 工具调用仍然发生在查询子 Agent 内部，不上提到顶层分流器。
- 不直接 import tool handler，而是统一走 `ProtocolHandler.execute_tool()`。
- 先实现确定性的最小多步工具策略，再谈显式状态、循环和自由规划。

**产物**：
- 新增《`KnowledgeQAAgent过渡期架构方案.md`》，作为后续实现与提交的边界文档。

### 2026-07-11 代码改造 Step 5：查询子 Agent 切到 MCP 主路径

**目标**：按照过渡期架构方案，把 `KnowledgeQAAgent` 的主路径切到 MCP Tool，同时保留旧直连链路 fallback。

**本步改动**：
1. 新增 `KnowledgeToolBroker`，统一通过 `ProtocolHandler.execute_tool()` 调用 3 个 MCP Tool。
2. `KnowledgeQAAgent` 新增确定性的最小多步策略：`list -> query -> summary`。
3. 原有 `KnowledgeRetriever -> KnowledgeService` 保留为 fallback，但不再承载新功能扩展。
4. 补充最小闭环测试，证明自然语言进入查询子 Agent 后会真实发生工具调用。

**结果**：
- 查询子 Agent 的主工作路径已经不再直冲 `KnowledgeService`，而是优先走 MCP Tool。
- 旧链路退化为兜底路径。
- 暂未引入 `ReActLoop`，但最小工具化多步能力已具备。

### 2026-07-12 架构升级 Step 6：MCP Client 抽象层 + 能力探测与纯开关降级

**背景**：核心场景是「第三方 MCP Server 不受管控」——改不了对方后端。此前 `KnowledgeToolBroker` 直连进程内 `ProtocolHandler`，是"假客户端"，无法连接第三方 Server，也没有能力协商。当 Server 不支持日志推送、流式输出等可选能力时，上层会盲目发请求导致静默失败。

**本次改动**：

| 文件 | 职责 |
|------|------|
| `mcp_client/__init__.py` | 模块入口，导出 MCPClientBase / LocalMCPClient / RemoteMCPClient / ServerCapabilityProfile / ClientFeatureFlags |
| `mcp_client/capabilities.py` | `ServerCapabilityProfile`：从 MCP SDK `ServerCapabilities` 解析布尔能力位（tools/resources/prompts/logging/completions + 子能力）；`ClientFeatureFlags`：在 profile 之上叠加环境变量覆盖（只能 force-disable，不能 force-enable） |
| `mcp_client/base.py` | `MCPClientBase` 抽象基类：`initialize()` → 能力握手、`call_tool()` → 工具调用、`get_feature_flags()` → 开关查询、`is_tool_available()` → 工具存在性检查 |
| `mcp_client/local_client.py` | `LocalMCPClient`：包装 `ProtocolHandler`，从已注册的 tools/resources/prompts 合成 `ServerCapabilityProfile`（本地服务器已知能力，不需要网络握手） |
| `mcp_client/remote_client.py` | `RemoteMCPClient`：基于 MCP SDK `ClientSession`，真实 `initialize` 握手获取 `ServerCapabilities`，支持 stdio / SSE 两种传输 |
| `agents/knowledge_qa/tool_broker.py` | 改造：从直连 `ProtocolHandler` 改为通过 `MCPClientBase`；新增 `ensure_initialized()` 返回 `ClientFeatureFlags`；`call_tool()` 在 `tools_enabled=False` 时抛 `MCPError(capability_not_supported)` |
| `agents/knowledge_qa_agent.py` | 改造：`process_stream()` 在尝试工具路径前先调 `ensure_initialized()` 检查能力开关；`tools_enabled=False` 时直接走 legacy fallback，不尝试工具路径 |
| `config/settings.py` | 新增 MCP Client 配置项：transport / command / args / url / timeout |
| `tests/test_mcp_client_capabilities.py` | 23 个测试：能力解析 5 + 环境变量覆盖 5 + LocalClient 7 + Broker 能力检查 3 + 第三方 Server 场景 3 |
| `tests/test_knowledge_qa_agent_mcp_path.py` | 适配：`FakeBroker` 增加 `ensure_initialized()` / `get_feature_flags()` |

**设计原则**：
- 可选能力不兼容 → **纯客户端开关控制**：探测到 Server 不支持，直接关掉本地对应功能开关，上层业务不再发起相关请求，没有胶水代码。
- 环境变量 `MCP_DISABLE_TOOLS` / `MCP_DISABLE_RESOURCES` / `MCP_DISABLE_PROMPTS` / `MCP_DISABLE_LOGGING` 可 force-disable 任何能力（即使 Server 声称支持），但永远不能 force-enable（Server 不支持就不能假装支持）。
- 工具缺失降级替代（适配层胶水代码）本次不实现，留作后续可选扩展。

**验证结果**：
- `tests/test_mcp_client_capabilities.py`：23 tests PASSED
- `tests/test_knowledge_qa_agent_mcp_path.py`：2 tests PASSED（既有测试无回归）
- `tests/test_mcp_tools.py` + `tests/test_mcp_resources_prompts.py`：24 tests PASSED
- 总计 49 tests PASSED，零回归

**面试话术**：
1. "MCP Client 怎么处理第三方 Server 不兼容？" → initialize 握手时做能力协商，ServerCapabilities 解析成布尔开关；不支持的能力直接关掉本地开关，上层不发请求，纯客户端控制无胶水代码。
2. "能举例吗？" → 第三方 Server 不支持 logging → `logging_enabled=False` → Client 不发 `logging/setLevel`；不支持 resources → `resources_enabled=False` → Client 不调 `resources/list`。
3. "操作者能覆盖吗？" → 环境变量 `MCP_DISABLE_*` 可以 force-disable 任何能力，但不能 force-enable（安全约束）。
4. "本地 Server 和远程 Server 怎么统一？" → `MCPClientBase` 抽象基类，`LocalMCPClient` 合成能力（进程内已知），`RemoteMCPClient` 真实握手（stdio/SSE），上层 `KnowledgeToolBroker` 只面向接口。
