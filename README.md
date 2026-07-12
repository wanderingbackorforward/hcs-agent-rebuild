# HCS 测试辅助 Agent 平台

> 二次实现版：基于 `smart-appointment-ai-agent` 的多 Agent 架构与 `MODULAR-RAG-MCP-SERVER` 的 RAG + MCP 能力，从零搭建一个可运行的 HCS（混合云）测试辅助 Agent 平台。

---

## 项目目标

让测试人员（尤其是新人）通过自然语言对话完成：
1. **环境条件确认**：根据测试意图自动补全环境类型、组件需求、服务状态等字段，筛选候选环境。
2. **技术规范查询**：通过 MCP 化的知识库检索 SDK 文档、用户手册、内部测试规范。

---

## 核心能力

| 能力 | 说明 |
|------|------|
| 多 Agent 协作 | TaskClassificationAgent → EnvironmentMatchingAgent / KnowledgeQAAgent |
| 三层意图路由 | L1 关键词切换 → L2 模糊/多诉求拦截 → L3 上下文锁定续接 |
| NLI 置信终审 | LLM 路由后用 NLI 校验 query↔职责描述匹配度，可插拔降级 |
| SSE 流式协议 | token / status / decision / error / done 五类事件，前端实时渲染 |
| 决策可解释性 | 每步路由决策生成用户可见解释，安全字段白名单过滤 |
| 任务中断与检查点 | 协作式取消 + pipeline 检查点，支持任务恢复 |
| 错误分类与重试 | 异常自动分类（可重试/不可重试），max-retry 保护 |
| Agent Memory | 会话级上下文补全，持续抽取环境条件字段 |
| MCP 知识检索 Server | 暴露 `query_knowledge_hub` / `list_collections` / `get_document_summary` |
| Hybrid Search | Dense Embedding + BM25 稀疏检索 + RRF 融合 + 可选 Rerank |
| Chunk 质量守卫 | 规则层 + 语义重切分，过滤低质量分块 |
| 文档摄取流水线 | Load → Split → Transform → Embed → Upsert |
| 环境匹配 | 结构化字段筛选 + 模拟 Linux 节点探测 + SQLite 验证快照 |
| 意图路由准确率评估 | 55 条 Golden Test + 准确率/Precision/Recall/F1 + 混淆矩阵 |
| 线上负样本收集 | 用户否定路由 + NLI 校验退回，定期更新 Few-Shot |
| 离线/线上偏差监控 | 离线评估 vs 线上 snapshot 对比，预期偏差 5-10% |
| TDD / OpenAPI | 单元测试覆盖 + FastAPI 自动生成接口文档 |

---

## 技术栈

- **Web 框架**：FastAPI + Uvicorn
- **AI 框架**：LangChain
- **LLM/Embedding**：OpenAI-compatible API（支持 Qwen / DeepSeek / Zhipu / Azure）
- **向量库**：ChromaDB
- **稀疏检索**：自研 BM25
- **数据库**：SQLite + SQLAlchemy
- **缓存/限流**：Redis（可选，用于分布式限流和 LLM 结果缓存）
- **MCP 协议**：mcp SDK（stdio）
- **前端**：Jinja2 + 原生 JS（可选）

---

## 目录结构

```
hcs-agent-rebuild/
├── app.py                          # FastAPI 应用入口
├── requirements.txt
├── .env.example
├── README.md
├── config/                         # 配置层
│   ├── model_provider.py           # 多模型 Provider 工厂
│   ├── database.py                # 数据库与 Redis 配置
│   ├── constants.py               # 信号词、切换词、否定词、阈值
│   ├── settings.py                # 含 ENABLE_NLI / NLI 阈值 / 置信度阈值
│   ├── audit.py                   # 全链路审计日志
│   ├── error_classifier.py        # 异常分类 + 重试决策
│   ├── decision_explainer.py      # 决策解释 + 安全字段白名单
│   └── sse_protocol.py            # SSE 事件协议（5 类事件）
├── db/                             # 数据持久层
│   ├── models.py
│   ├── base.py
│   ├── db_router.py
│   └── repositories/
├── agents/                         # Agent 智能层
│   ├── task_classification_agent.py  # 主控制器，装配 NLI + SemanticChecker
│   ├── task_classification/
│   │   ├── task_classifier.py      # LLM 意图分类器
│   │   ├── classification_processor.py  # 三层网关 + NLI 终审 + 负样本收集
│   │   ├── agent_router.py         # 路由分发器
│   │   ├── nli_validator.py        # NLI 置信校验器（可插拔，降级）
│   │   ├── semantic_checker.py     # 语义续接检查器
│   │   ├── state_manager.py        # 状态管理器
│   │   ├── json_utils.py           # 分类 JSON 解析
│   │   └── unrelated_handler.py
│   ├── environment_matching_agent.py
│   ├── environment_matching/
│   ├── knowledge_qa_agent.py
│   ├── memory/                     # 分层记忆（STM/LTM/TaskMemory）
│   ├── context_lock.py            # 上下文锁定持久化
│   └── task_manager.py             # 任务管理器（取消 + 检查点）
├── services/                       # 业务服务层
│   ├── environment_service.py
│   ├── probe_service.py
│   └── knowledge_service.py
├── mcp_server/                     # MCP 知识检索 Server
│   ├── server.py
│   ├── protocol_handler.py
│   ├── prompts/                    # classify-intent 等 MCP Prompt
│   └── tools/
├── rag/                            # RAG 检索与摄取
│   ├── ingestion/
│   │   └── chunking/
│   │       └── quality_guard.py   # 分块质量守卫
│   └── query_engine/
├── eval/                           # 评估框架
│   ├── intent_routing_eval.py     # 55 条 Golden Test + P/R/F1 + 混淆矩阵
│   ├── golden_cases.py            # Agent 级 golden 标注 + 回放 trace
│   ├── offline.py                 # 离线评估器（replay/live 双模式）
│   ├── online.py                  # 线上评估器 + 负样本收集 + 离线对比
│   ├── metrics.py                 # 5 指标聚合
│   ├── metrics_content.py         # 任务成功率 + 回答质量/幻觉检测
│   ├── metrics_process.py        # 执行轨迹 + 工具调用
│   ├── metrics_cost.py           # P50/P95 延迟 + Token 成本
│   ├── trace.py                   # Trace 数据模型
│   └── report.py                  # Markdown 报告渲染
├── api/                            # API 层
│   ├── chat_handler.py
│   └── core/
├── web/                            # Web 层
│   ├── routes.py
│   ├── templates/
│   └── static/
├── prompts/                        # LLM Prompt 模板
│   ├── classification_v1.txt      # 意图分类 Prompt
│   ├── context_lock_judge_v1.txt   # 上下文续接判定
│   └── eval_llm_judge_v1.txt      # LLM Judge Prompt
├── docs/                           # 文档
│   ├── intent_routing_eval_report.md  # 意图路由评估报告
│   └── agent_eval_report_sample.md
└── tests/                          # 测试（22 个文件）
    ├── test_task_classification_agent.py  # 4 层测试 + 55 Golden Cases
    ├── test_nli_validator.py              # NLI + 降级 + 否定检测
    ├── test_eval_framework.py            # 5 指标评估框架
    ├── test_decision_explainer.py
    ├── test_error_classifier.py
    ├── test_context_lock.py
    ├── test_sse_protocol.py
    ├── test_task_manager.py
    ├── test_hybrid_search.py
    └── ...
```

---

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 LLM API Key 与 Base URL
```

### 3. 运行

```bash
python app.py
```

访问 http://localhost:8000/docs 查看 API 文档。

---

## 意图路由与 NLI 置信终审

```
User Query
  → L1: 关键词切换检测 (SWITCH_WORDS)
  → L1.5: 用户否定检测 (REJECTION_WORDS → 记录负样本)
  → L2: 模糊/多诉求拦截 (confidence pre-check)
  → L3: 上下文锁定续接 (SemanticChecker)
  → LLM 分类: TaskClassifier → intent_type + confidence
  → NLI 终审: NLIValidator.nli_check(query, intent_type)
      ├─ score ≥ 0.7  → 放行
      ├─ 0.6 ~ 0.7    → 谨慎放行
      ├─ < 0.6        → 兜底回复 + 记录负样本
      └─ None         → 降级: LLM confidence ≥ 0.8 + 关键词命中
```

NLI 可通过 `ENABLE_NLI=False` 一键关闭，或 embedder 不可用时自动降级，零影响原有链路。

---

## 文档摄取与分块策略

```
原始文档
  → 主切分器: RecursiveCharacterTextSplitter (字符级, 600字符/100重叠, 全程无 embedding)
  → ChunkQualityGuard 中间件 (默认关闭, 启用后执行以下两步)
      ① 规则前置筛选 (零 embedding, 纯字符扫描)
          规则A structure_starved: chunk 长度 ≥ 阈值但分隔符(换行/句号)少于阈值 → 可疑
          规则B multi_section:   单个 chunk 命中 ≥2 个章节关键词(第X章/##/1./一、) → 可疑
          未命中规则 → 直接判定合格, 跳过 embedding
      ② 语义再切 (仅对可疑 chunk)
          句级切分 → embed_batch 每句 → 相邻 cosine 相似度 → 低于阈值处切分
          embedder 出错则降级保留原 chunk
  → Embedder: 对最终 chunks 全量 embed
  → ChromaStore: upsert
```

**设计要点**：主切分始终是字符级 `RecursiveCharacterTextSplitter`，语义切分仅作为可疑 chunk 的可选兜底，避免"全文档语义切分"的高开销反模式。默认关闭（`CHUNK_GUARD_ENABLED=False`），小文档场景零额外成本；接入几十万字大文档时开启。

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `CHUNK_GUARD_ENABLED` | `False` | 总开关，关闭时 guard 为纯 pass-through |
| `CHUNK_GUARD_MIN_CHARS` | `300` | 触发结构检查的最小字符数，短于此不检查 |
| `CHUNK_GUARD_MIN_SEPARATORS` | `2` | 合格 chunk 至少含有的分隔符数，低于此值且超长则可疑 |
| `CHUNK_GUARD_RESPLIT_THRESHOLD` | `0.45` | 语义再切的相邻句相似度阈值，低于此值在该处切分 |

相关代码：`rag/ingestion/chunking/quality_guard.py`、`rag/ingestion/pipeline.py`、`config/chunker_factory.py`

---

## Embedding 模型选型

本平台面向华为 SDK 及专业技术文档的检索场景，Embedding 模型选型遵循"先卡硬要求、再选性价比维度、最后对照模型清单"的三步法。

### 选型三步法

**第一步：两个硬要求（不满足直接 pass）**

1. **中文优化的检索专用模型**：场景是"用户问短问题，搜长的技术文档"，通用句向量模型和英文模型在中文技术术语上语义匹配能力不足，选了必崩。
2. **覆盖技术语料训练**：SDK 文档里全是接口名、参数定义、示例代码，未经技术语料训练的模型对这些内容的语义理解很差，匹配不准。

**第二步：维度选 1024 维**

| 维度 | 适用判断 | 选型结论 |
|------|---------|---------|
| 768 | 太浅，区分不开相似接口和参数细节（如 `init()` 与 `initWithConfig()`） | 不够 |
| 1024 | 覆盖技术文档细粒度语义，成本和检索速度可控 | **当前选型** |
| 1536/3072 | 效果只提升一点点，存储翻倍、检索明显变慢 | 性价比极低 |

维度不是越高越好——超过 1024 维后收益边际递减，到 2048+ 甚至出现"维度灾难"，相似度计算反而不稳。

**第三步：模型清单（直接抄作业）**

| 部署方式 | 模型 | 维度 | 说明 |
|---------|------|------|------|
| 云端 API | MiniMax `embo-01` | 1024 | 当前默认，新手友好 |
| 云端 API | 通义千问 `text-embedding-v3` | 1024 | OpenAI 兼容接口 |
| 本地部署 | `BAAI/bge-large-zh-v1.5` | 1024 | 中文技术文档标杆 |
| 本地部署 | `BAAI/bge-m3` | 1024 | 带代码匹配效果更好 |

### 配置方式

通过环境变量切换，不改代码：

```bash
# 云端 API（默认）
EMBEDDING_PROVIDER=minimax
EMBEDDING_MODEL=embo-01

# 内网私有化部署（local provider，需安装 sentence-transformers）
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
EMBEDDING_DEVICE=cpu
```

### 选错模型的现象

- **搜不到正确内容**：用户问"SDK 怎么设置请求超时"，结果搜出安装说明——模型看不懂技术术语，语义匹配错位
- **相似接口分不清**：两个功能接近的 SDK 接口，模型区分不开，把错误参数说明排到最前面
- **代码查询失效**：用户贴报错代码或搜示例代码完全匹配不上——模型没经过代码训练，看不懂代码语义
- **成本暴涨检索变慢**：盲目选 3072 维，文档量上来后单次查询等好几秒，API 费翻倍但效果几乎没提升

相关代码：`config/model_provider.py`、`rag/ingestion/embedding/embedder.py`

---

## 评估体系

### 离线评估

- **55 条 Golden Test**：25 environment_match + 25 knowledge_qa + 5 unrelated
- **指标**：准确率、各意图 Precision / Recall / F1、混淆矩阵
- **模式**：replay（CI-safe，无需 API Key）+ live（真实 LLM）
- **当前结果**：准确率 94.5%（live LLM）

```bash
python -m pytest tests/test_task_classification_agent.py -v
```

### 线上评估

- **负样本收集**：用户否定路由（`REJECTION_WORDS`）+ NLI 校验退回
- **端点**：`GET /eval/metrics`、`GET /eval/traces`、`GET /eval/negative-samples`
- **离线/线上偏差**：`snapshot.gap` 字段，预期离线比线上高 5-10%

---

## 新增意图类别扩展指南

当 Agent 数量从 2 个扩展到更多时，按以下 4 步操作，已有意图路由不受影响。

### 扩展步骤

1. **Prompt 加新类别描述**：在 `prompts/classification_v1.txt` 的 `intent_type` 取值中新增类别，附 2-3 条示例
2. **新建子 Agent**：在 `agents/` 下新建 Agent 模块，在 `agent_router.py` 加路由分支，在 `decision_explainer.py` 加显示名
3. **NLI 描述注册**：在 `nli_validator.py` 的 `AGENT_DESCRIPTIONS` 加新意图的职责描述
4. **Golden Test 补充**：在 `eval/intent_routing_eval.py` 的 `GOLDEN_CASES` 加新意图用例，`INTENTS` 元组加新标签

### 测试重点（按优先级）

| 优先级 | 测试类型 | 做什么 | 防什么 |
|--------|---------|--------|--------|
| P0 | **边界混淆用例** | 补新旧意图与老意图职责重叠的模糊 case（如"慢查询日志"既沾数据库又沾日志分析） | 新类别抢老意图流量、老意图错进新类别 |
| P1 | **新意图召回用例** | 补新意图的口语化、长尾变体表达，验证识别率 | 加了新类别但识别不出来，全漏去老意图 |
| P2 | **老意图全量回归** | 跑通原有 55 条 Golden Test，确认准确率无下降 | 加新类别带崩原有路由效果 |

```bash
# 跑全量回归（含新意图用例）
python -m pytest tests/test_task_classification_agent.py -v

# 只看老意图是否受影响（replay 模式，无需 API Key）
python -m pytest tests/test_task_classification_agent.py::TestRoutingMetricsReplay -v
```

### 多 Agent 扩容路径

| Agent 数量 | 架构 | 改动 |
|-----------|------|------|
| 2-5 | 当前方案直接适用 | 只需 4 步扩展 + Golden Test |
| >5 | Step 4 前插入 Embedding 粗筛层 | 预生成所有 Agent 职责 Embedding，用 query 召回 Top-3 候选，再走 LLM 选型 + NLI 校验 |

NLI 校验器已预留多 Agent 接口：`AGENT_DESCRIPTIONS` 字典加条目即可，`nli_check()` 自动处理。

---

## 当前状态

- [x] 项目骨架
- [x] 配置层
- [x] 数据层
- [x] RAG 模块（含 Chunk 质量守卫）
- [x] MCP Server
- [x] Agent 层（含 NLI 置信终审）
- [x] API / Web 层
- [x] SSE 流式协议
- [x] 决策可解释性
- [x] 任务中断与检查点
- [x] 错误分类与重试
- [x] 意图路由准确率评估（Golden Test + F1 + 负样本收集）
- [x] 测试
- [x] 文档
- [ ] Redis 集成（分布式限流、LLM 结果缓存）
