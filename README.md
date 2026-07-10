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
