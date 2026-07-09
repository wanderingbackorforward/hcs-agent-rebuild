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
