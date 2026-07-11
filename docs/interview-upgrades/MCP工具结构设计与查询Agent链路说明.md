# MCP 工具结构设计与查询 Agent 链路说明

## 本文目的

当前项目已经具备顶层分流、知识查询子 Agent、MCP Server、3 个 MCP Tool、Hybrid Search、缓存与错误封装等能力，但这几部分在表达和链路上容易被混淆。本文只做两件事：

1. 重新整理 3 个 MCP Tool 的参数与返回结构设计，让工具对 Agent 更可消费。
2. 写清当前查询子 Agent 的真实执行链路，明确哪些部分已经打通，哪些部分还没有打通。

本文不讨论是否接入 `ReActLoop`，也不改动顶层分流逻辑。

## 当前真实链路

当前用户自然语言请求进入系统后的主链路如下：

```text
用户输入
  -> api/chat_handler.py
  -> TaskClassificationAgent.classify_task_stream()
  -> ClassificationProcessor.process_task_stream()
  -> AgentRouter.route()
  -> KnowledgeQAAgent.process_stream()
  -> KnowledgeRetriever.retrieve()
  -> KnowledgeService.search()
  -> HybridSearch.search()
  -> DenseRetriever + SparseRetriever + RRF + reranker
  -> KnowledgeQAAgent 内部生成答案
```

这条链路里，顶层分流和子 Agent 分工是清楚的：

- `TaskClassificationAgent` 负责把用户问题分到正确子 Agent。
- `KnowledgeQAAgent` 负责处理知识查询这一类请求。
- `KnowledgeQAAgent` 当前使用的是内部服务链路，而不是 MCP Tool 调用链路。

## 当前架构中 3 个 MCP Tool 的位置

当前 3 个 MCP Tool 独立存在于 `mcp_server/tools/`：

- `query_knowledge_hub`
- `list_collections`
- `get_document_summary`

它们通过 `mcp_server/protocol_handler.py` 注册到 MCP Server 中，当前可以被外部 MCP Client 调用，也可以作为后续内部工具化改造的能力基础。

但在当前生产链路中，`KnowledgeQAAgent` 并没有通过 `protocol_handler.execute_tool()` 去调用这 3 个 Tool，而是直接走了：

```text
KnowledgeQAAgent -> KnowledgeRetriever -> KnowledgeService -> HybridSearch
```

所以当前最准确的判断是：

- 顶层自然语言分流已经打通。
- 查询子 Agent 已经打通。
- MCP Server 及 3 个 Tool 已经实现。
- 但“查询子 Agent 在生产链路中真实通过 MCP Tool 工作”这件事还没有接上线。

## 当前 3 个 Tool 的问题

### `query_knowledge_hub`

当前问题主要有三类：

1. 参数少，无法显式控制返回形态。
2. 返回值是 Markdown 文本，结构化信息不足，Agent 二次消费困难。
3. 检索结果、生成答案、缓存命中、降级情况混在一起，不利于调试与编排。

### `list_collections`

当前问题主要有两类：

1. 只有 `include_stats` 一个参数，筛选和视角都太单薄。
2. 返回值是面向人读的列表文本，不利于 Agent 做下一步决策。

### `get_document_summary`

当前问题主要有三类：

1. 只能按 `doc_id` 查摘要，粒度较粗。
2. 元数据不完整，缺少标题、类别、来源等明确结构。
3. 返回仍然是 Markdown 文本，Agent 若要继续精读或做判断，需要额外解析。

## 工具设计原则

本轮不新增 Tool，只优化已有 3 个 Tool 的设计。统一遵循以下原则：

- 面向 Agent 消费优先，不再只面向人类阅读。
- 保留简洁默认值，但允许显式控制返回粒度。
- 返回结构中同时包含：业务结果、元数据、执行信息、降级信息。
- 错误仍然沿用当前 `FormattedError` 风格，避免泄露内部堆栈。

## Tool 1 设计

### `query_knowledge_hub`

#### 建议定位

它不只是“查知识库”，而是“统一的知识查询入口”。既要支持直接给最终答案，也要支持返回原始证据，便于后续 Agent 做二次决策。

#### 建议输入参数

```json
{
  "query": "string, 必填，用户问题或检索语句",
  "top_k": "integer, 可选，默认 5，范围 1-20",
  "collection": "string, 可选，集合或逻辑命名空间",
  "category": "string, 可选，按 sdk/spec/manual 等类别过滤",
  "doc_id": "string, 可选，限定在单文档内检索",
  "return_mode": "string, 可选，answer | chunks | both，默认 both",
  "include_scores": "boolean, 可选，默认 true",
  "include_metadata": "boolean, 可选，默认 true",
  "max_chars_per_chunk": "integer, 可选，默认 300"
}
```

#### 建议输出结构

```json
{
  "query": "用户原始查询",
  "answer": "可为空；当 return_mode 包含 answer 时返回",
  "retrieved_chunks": [
    {
      "rank": 1,
      "doc_id": "hcs-sdk-quickstart",
      "title": "HCS SDK 快速入门",
      "category": "sdk",
      "source": "seed",
      "chunk_index": 0,
      "score": 0.8123,
      "text_preview": "HCS SDK 是华为混合云平台的开发工具包……"
    }
  ],
  "filters_used": {
    "collection": "可为空",
    "category": "可为空",
    "doc_id": "可为空"
  },
  "top_k_used": 5,
  "cache_hit": false,
  "answer_generated_by": "llm | fallback",
  "result_count": 3,
  "trace_id": "错误或调试链路追踪 ID"
}
```

#### 设计价值

- 对人类用户：仍可直接拿到答案。
- 对 Agent：可以明确拿到命中文档、分数、类别、来源和片段预览。
- 对后续调试：能区分缓存命中、LLM 成功生成、检索命中但生成失败等状态。

## Tool 2 设计

### `list_collections`

#### 建议定位

它不应该只是列集合名，而应该承担“发现知识空间结构”的职责，帮助 Agent 在查询前先理解可用范围。

#### 建议输入参数

```json
{
  "include_stats": "boolean, 可选，默认 true",
  "include_categories": "boolean, 可选，默认 true",
  "include_doc_samples": "boolean, 可选，默认 false",
  "sample_size": "integer, 可选，默认 3",
  "keyword": "string, 可选，按标题或类别关键字过滤"
}
```

#### 建议输出结构

```json
{
  "collections": [
    {
      "name": "hcs_knowledge",
      "doc_count": 3,
      "categories": [
        {
          "name": "sdk",
          "doc_count": 1
        },
        {
          "name": "spec",
          "doc_count": 1
        },
        {
          "name": "manual",
          "doc_count": 1
        }
      ],
      "sample_docs": [
        {
          "doc_id": "hcs-sdk-quickstart",
          "title": "HCS SDK 快速入门",
          "category": "sdk"
        }
      ]
    }
  ],
  "total_collections": 1,
  "keyword_used": "可为空"
}
```

#### 设计价值

- 让 Agent 先知道“知识库里有什么”，再决定是否直接检索。
- 为后续如果接入工具式决策留出空间。
- 对面试表述更合理，因为它体现的是“知识发现”而不是“打印字符串”。

## Tool 3 设计

### `get_document_summary`

#### 建议定位

它应该是“文档级查看工具”，用于在已经定位到文档后，快速了解其摘要与元信息，而不是简单按字符截断。

#### 建议输入参数

```json
{
  "doc_id": "string, 必填",
  "collection": "string, 可选",
  "max_chars": "integer, 可选，默认 500",
  "include_metadata": "boolean, 可选，默认 true",
  "include_source": "boolean, 可选，默认 true",
  "include_chunk_stats": "boolean, 可选，默认 true"
}
```

#### 建议输出结构

```json
{
  "doc_id": "hcs-sdk-quickstart",
  "title": "HCS SDK 快速入门",
  "collection": "hcs_knowledge",
  "category": "sdk",
  "source": "seed",
  "summary": "HCS SDK 是华为混合云平台的开发工具包……",
  "content_length": 1260,
  "chunk_count": 4,
  "metadata": {
    "title": "HCS SDK 快速入门",
    "source": "seed"
  },
  "trace_id": "错误或调试链路追踪 ID"
}
```

#### 设计价值

- 让 Agent 不用解析 Markdown 文本就能拿到文档关键信息。
- 为未来如果新增精读型工具打基础。
- 面向人类和面向机器都更稳定。

## 当前链路里已经清楚的职责分工

### 顶层分流

由 `TaskClassificationAgent` 完成。它的职责是回答“这条请求应该交给哪个子 Agent”。

### 查询子 Agent

由 `KnowledgeQAAgent` 完成。它的职责是回答“知识问答请求已经进入本 Agent 后，应该怎么检索和怎么生成答案”。

### MCP Server

由 `mcp_server/` 完成。它当前承担的是“把知识能力标准化暴露成 Tool / Resource / Prompt”的职责。

## 当前链路里真正的断点

真正的断点不在顶层分流，而在查询子 Agent 与 MCP Tool 之间。

现在的情况不是“系统不会分流到查询子 Agent”，而是：

- 已经能分流到 `KnowledgeQAAgent`
- 但 `KnowledgeQAAgent` 还没有在主链路中去调用这 3 个 MCP Tool

换句话说，当前缺的不是“再加一个 Agent”，而是“是否让查询子 Agent 内部真正工具化”。

## 本轮结论

当前最稳妥的推进顺序应该是：

1. 先把 3 个 MCP Tool 的结构设计好，避免后续接线时 schema 再返工。
2. 先把查询子 Agent 当前链路说清楚，避免把顶层分流和内部执行模式混在一起。
3. 在这两个前提清楚后，再决定是否需要把 `ReActLoop` 作为 `KnowledgeQAAgent` 的内部执行模式接入。

## 面试可直接使用的话术

我项目当前已经实现了顶层自然语言分流、多子 Agent 架构、知识查询子 Agent、MCP Server 和 3 个知识工具。现在主链路里，用户问题会先被分流到 `KnowledgeQAAgent`，再由内部的知识服务链路完成 Hybrid Search 和答案生成。MCP 侧的 3 个工具已经具备能力基础，但为了让后续查询子 Agent 真正工具化，我优先把这 3 个 Tool 的参数和返回结构重新设计成更适合 Agent 消费的形式，再决定是否把工具式执行策略接入主链路。这样做的好处是，不会把顶层分流逻辑和子 Agent 内部执行策略混在一起，改造路径也更稳定。
