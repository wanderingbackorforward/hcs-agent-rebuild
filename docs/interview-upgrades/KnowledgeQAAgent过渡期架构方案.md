# KnowledgeQAAgent 过渡期架构方案

## 目的

当前项目已经完成 3 个 MCP Tool 的结构化升级，但 `KnowledgeQAAgent` 的主链仍然是直连 `KnowledgeService`。本方案用于明确过渡期版本的架构边界，避免一边实现一边摇摆。

本文只回答 4 个问题：

1. 查询子 Agent 的主路径是什么。
2. 旧直连链路在过渡期还承担什么职责。
3. 查询子 Agent 最小要具备哪些多步能力。
4. 当前阶段明确不做什么。

## 当前状态

当前知识问答链路是：

```text
用户输入
  -> TaskClassificationAgent
  -> KnowledgeQAAgent
  -> KnowledgeRetriever
  -> KnowledgeService
  -> HybridSearch
  -> LLM 生成答案
```

与此同时，MCP 侧已经有 3 个可直接调用的工具：

- `list_collections`
- `query_knowledge_hub`
- `get_document_summary`

也就是说，当前系统具备两套能力：

- 一套是查询子 Agent 内部的直连检索链路
- 一套是标准化暴露出来的 MCP Tool

真正缺的不是再补一个 Tool，而是让查询子 Agent 的主工作路径切到 MCP Tool。

## 过渡期目标

过渡期版本不追求一步到位做成完整 Agent，而是先完成以下 5 件事：

1. `KnowledgeQAAgent` 的主路径切到 MCP Tool。
2. 原有 `KnowledgeRetriever -> KnowledgeService` 保留为 fallback。
3. 查询子 Agent 至少具备最小多步策略：`list -> query -> summary`。
4. 暂不上 `ReActLoop`，不做自由工具规划。
5. 不再给旧直连链路增加新能力。

## 最终判断

过渡期应采用下面这条路线，而不是直接全量改成 ReAct Agent：

```text
TaskClassificationAgent
  -> KnowledgeQAAgent
       -> MCP Tool 主路径
       -> 旧直连链路 fallback
```

这是一条过渡架构，不是终态架构。  
终态仍然是逐步废弃旧直连链路，但当前阶段先不直接删除。

## 主路径

### 方案结论

过渡期主路径定为：

```text
KnowledgeQAAgent
  -> KnowledgeToolBroker / MCP Tool Client
  -> ProtocolHandler.execute_tool(...)
  -> query_knowledge_hub / list_collections / get_document_summary
```

这里有两个关键点：

1. 工具调用发生在 `KnowledgeQAAgent` 内部，不上提到顶层分流器。
2. 不直接 import 某个 tool handler，而是统一通过 `ProtocolHandler.execute_tool()` 调用。

### 为什么不能直接 import handler

如果查询子 Agent 直接调用：

- `query_knowledge_hub_handler(...)`
- `list_collections_handler(...)`
- `get_document_summary_handler(...)`

虽然也能运行，但它绕开了统一的 MCP Tool 执行入口，丢掉了下面这些能力：

- 统一审计
- RBAC 检查
- 一致的错误包装
- 后续工具轨迹统计

所以过渡期主路径虽然不是“远程真正的 MCP Client”，但也必须通过 `ProtocolHandler.execute_tool()` 进入工具层，不能绕开协议层。

## fallback

### 方案结论

原有直连链路保留，但只作为 fallback：

```text
KnowledgeQAAgent
  -> MCP Tool 主路径
  -> 若主路径失败，再降级到
     KnowledgeRetriever -> KnowledgeService
```

### fallback 的职责

fallback 在过渡期只负责两类情况：

1. 工具调用失败，例如返回错误或结果为空且不满足回答要求。
2. 新主路径还没覆盖的特殊情况，需要旧链路兜底保持功能可用。

### fallback 的边界

从现在开始，不再继续往旧直连链路里加新能力。  
也就是说：

- 不给 `KnowledgeRetriever` 增加新的工具逻辑
- 不给 `KnowledgeService.search()` 增加新的 Agent 决策职责
- 旧链路只负责稳定兜底

这样才能保证它最终能被平滑废弃，而不是越改越重。

## 最小多步策略

当前阶段不上完整 ReAct，但查询子 Agent 不能只是“单问单工具”。  
过渡期至少要具备下面这套确定性的多步策略。

### 策略 1：发现优先

当用户在问：

- 有哪些文档
- 有哪些类别
- 知识库里有什么
- SDK / 手册 / 规范分别有哪些

优先走：

```text
list_collections
```

如果用户问题里还带了类别提示词，例如“看看 SDK 相关文档”，则把关键词传给 `list_collections.keyword`。

### 策略 2：一般知识问答

当用户在问一般知识问题，例如：

- HCS 是什么
- SDK 怎么初始化
- 测试规范里怎么要求环境隔离

优先走：

```text
query_knowledge_hub
```

如果 `query_knowledge_hub` 返回的结构里已经有足够答案，则直接汇总输出。  
如果答案为空、证据不够，才考虑下一步钻取。

### 策略 3：文档钻取

当用户明确提到了文档，或者上一步已经锁定某个文档时，走：

```text
get_document_summary
```

适用场景包括：

- “快速入门这篇文档讲什么”
- “帮我看一下 hcs-sdk-quickstart”
- “上一步命中的那篇 SDK 文档大意是什么”

### 最小链式动作

过渡期版本至少允许以下三种链路：

1. `list`
2. `query`
3. `list -> query`
4. `query -> summary`
5. `list -> query -> summary`

这意味着查询子 Agent 至少已经具备：

- 先发现知识空间
- 再做查询
- 最后钻取文档

虽然还没有完整状态机，但已经不是单薄的一步式工具调用。

## 暂不上 ReAct

### 原因

当前阶段不引入 `ReActLoop`，原因不是它没价值，而是现在引入会把两个问题混在一起：

1. 查询子 Agent 是否真的已经接到 MCP Tool
2. 查询子 Agent 是否已经具备复杂循环推理能力

现在应该先解决第一个问题，再逐步补第二个问题。

### 当前阶段不做的事

过渡期版本明确不做：

- 不上完整 `Thought -> Action -> Observation` 循环
- 不让 LLM 自由选择任意工具
- 不做显式状态机
- 不做失败恢复分支
- 不做多轮最大步数控制

这些都属于下一阶段的 Agent 化增强，而不是当前过渡期的交付目标。

## 新增模块建议

过渡期建议新增一个轻量模块，例如：

```text
agents/knowledge_qa/tool_broker.py
```

职责很简单：

- 初始化带默认工具注册的 `ProtocolHandler`
- 对外提供统一的 `call_tool(name, args)` 方法
- 把 `CallToolResult` 解析成查询子 Agent 可消费的结构

### 建议接口

建议接口保持简单：

```python
await broker.call_tool("list_collections", {...})
await broker.call_tool("query_knowledge_hub", {...})
await broker.call_tool("get_document_summary", {...})
```

这样 `KnowledgeQAAgent` 只负责：

- 判断当前应该走哪一步
- 收集工具返回
- 决定下一步是否继续
- 汇总最终答案

而不负责直接操作 MCP 协议对象。

## KnowledgeQAAgent 需要拆出的执行阶段

过渡期建议把 `KnowledgeQAAgent.process_stream()` 内部逻辑拆成下面 4 个阶段：

### 1. 查询分析

根据用户问题决定属于哪类：

- 发现型
- 一般查询型
- 文档钻取型

### 2. 工具计划

生成一个确定性的最小计划，例如：

- `["list_collections"]`
- `["query_knowledge_hub"]`
- `["query_knowledge_hub", "get_document_summary"]`

### 3. 工具执行

按顺序调用工具，拿到结构化结果。  
如果失败，转 fallback。

### 4. 答案汇总

根据工具结果生成最终输出，并记录必要的任务进度、缓存和会话历史。

## 过渡期验收标准

当下面 5 条都成立时，说明过渡期版本算完成：

1. `KnowledgeQAAgent` 的主路径已经通过 MCP Tool 工作。
2. 工具调用统一经过 `ProtocolHandler.execute_tool()`。
3. 查询子 Agent 至少具备 `list -> query -> summary` 的最小多步策略。
4. 原有直连链路仅作为 fallback，不再承担新功能扩展。
5. 测试能证明“自然语言进入查询子 Agent后，确实发生了工具调用”。

## 本阶段后的下一步

过渡期完成后，再进入下一阶段：

- 弱化并逐步移除旧直连链路
- 引入显式状态和循环
- 增加 observation 后再决策
- 评估是否正式接入 `ReActLoop`

到那时，查询子 Agent 才会从“工具驱动查询器”逐步演化成真正的多步 Agent。
