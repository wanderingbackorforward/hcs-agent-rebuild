# Agent UX 升级计划：延迟感知 + 决策可解释 + 任务可中断 + 错误恢复

> 2026-07-09 制定。四个独立 feature 分支，按依赖顺序开发。

---

## 现状分析

| 维度 | 当前状态 | 目标状态 |
|------|---------|---------|
| 流式输出 | `/chat/stream` 返回 text/plain 纯文本流，前端用非流式 `/chat` | SSE 协议，结构化事件（status/token/decision/error/done），前端 EventSource 消费 |
| 中间状态 | 无中间状态推送，用户等全部处理完才看到内容 | 实时推送处理阶段（"正在分类意图..."等），降低感知延迟 |
| 决策可解释 | 有 audit_event 日志（intent_decision/context_lock_hit），但不暴露给用户 | decision 事件推送到前端，折叠面板展示路由原因+置信度 |
| 任务中断 | 无任何中断机制，无 CancelledError 处理 | 支持 cancel 端点 + 断点续传 + 前端取消按钮 |
| 错误恢复 | 有分层异常处理和优雅降级，但无用户建议/重试按钮 | 错误分类（可重试/不可重试）+ 建议 + 一键重试 |

---

## 分支策略

四个独立 feature 分支，按依赖顺序开发，每个完成后合并回 main 再开下一个：

```
main (当前 910452a)
  ├── feature/sse-streaming        ← 第1个，SSE协议升级（基础设施）
  │     合并回 main
  ├── feature/decision-explainability  ← 依赖 SSE 的 decision 事件
  │     合并回 main
  ├── feature/task-interruptible   ← 依赖 SSE 的 error/done 事件
  │     合并回 main
  └── feature/error-recovery       ← 依赖 SSE 的 error 事件
        合并回 main
```

---

## Feature 1: SSE 流式协议升级（feature/sse-streaming）

### 问题
- Agent 全部处理完才返回内容，用户等待焦虑
- 前端调用非流式 `/chat` 端点，无法实时展示
- text/plain 纯文本流无法携带结构化元数据

### 方案

#### 1.1 SSE 事件协议设计（新文件 `api/sse_protocol.py`）

定义 5 种事件类型，每种用 `data: {json}\n\n` 格式：

| 事件类型 | 用途 | data 字段 |
|---------|------|-----------|
| `status` | 处理阶段状态 | `{stage, message, timestamp}` |
| `token` | 回复内容分片 | `{content, seq}` |
| `decision` | 路由决策信息 | `{intent_type, confidence, reason, agent}` |
| `error` | 错误信息 | `{error_type, message, retryable, suggestion, seq}` |
| `done` | 完成信号 | `{session_id, task_id, elapsed_ms}` |

设计要点：
- 每条事件带 `seq` 序列号，客户端检测断流
- `status` 事件不含敏感信息，只展示处理阶段
- `error` 事件区分可重试/不可重试（为 Feature 4 预留）
- `decision` 事件经过 `mask_sensitive()` 处理（为 Feature 2 预留）

#### 1.2 后端流式改造

修改 `web/routes.py::chat_stream_endpoint`：
- `media_type` 从 `text/plain` 改为 `text/event-stream`
- `token_generator()` 包装为 SSE 格式输出
- 增加 `Cache-Control: no-cache` 和 `X-Accel-Buffering: no` 头

修改 `api/chat_handler.py::process_user_input_stream`：
- 在管道各阶段注入 status 事件（通过新的 yield 机制）
- 返回 `SSEEvent` 对象而非裸字符串

修改 Agent 层（`classification_processor.py`, `agent_router.py`, specialist agents）：
- 在关键节点 yield status 事件
- `process_task_stream()` 在分类前 yield `status:classifying`
- `AgentRouter.route()` 在路由前 yield `status:routing:{intent_type}`
- specialist agent 在处理前 yield `status:processing:{agent_name}`

#### 1.3 前端 SSE 消费

修改 `web/templates/index.html`：
- 从 `fetch('/chat')` 改为 `new EventSource('/chat/stream')`
- `onmessage` 解析 SSE 事件类型
- status 事件 → 显示处理状态条（带 spinner 文字）
- token 事件 → 追加到回复区域
- error 事件 → 显示错误 + 重试按钮（Feature 4）
- done 事件 → 关闭状态条

#### 1.4 网络韧性

- 服务端：每个事件带 `seq` 序列号
- 客户端：`onerror` 检测断连，记录最后 `seq`，自动重连并携带 `Last-Event-ID` 头
- 服务端：维护最近 50 条事件的环形缓冲区，支持 `Last-Event-ID` 续传

### 新问题应对：分片推送服务器压力 + 网络异常内容错乱

| 问题 | 应对 |
|------|------|
| 分片推送增加服务器压力 | 事件批量合并（50ms 内的事件合并为一个 SSE 帧），减少 HTTP 帧数 |
| 网络异常内容错乱 | seq 序列号 + Last-Event-ID 续传 + 客户端 seq 校验丢弃乱序事件 |
| 连接泄漏 | 服务端 heartbeat（每 15s 发 `:keepalive\n\n`），客户端 30s 超时断连重连 |

### 新增/修改文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `api/sse_protocol.py` | 新建 | SSE 事件类型定义 + 格式化函数 |
| `api/sse_buffer.py` | 新建 | 环形缓冲区，支持 Last-Event-ID 续传 |
| `web/routes.py` | 修改 | SSE 端点、Cache-Control 头 |
| `api/chat_handler.py` | 修改 | 管道注入 status 事件 |
| `agents/task_classification/classification_processor.py` | 修改 | yield status 事件 |
| `agents/task_classification/agent_router.py` | 修改 | yield status 事件 |
| `agents/environment_matching_agent.py` | 修改 | yield status 事件 |
| `agents/knowledge_qa_agent.py` | 修改 | yield status 事件 |
| `web/templates/index.html` | 修改 | EventSource + 状态条 + 流式展示 |
| `tests/test_sse_protocol.py` | 新建 | SSE 事件格式化测试 |

---

## Feature 2: 决策可解释性（feature/decision-explainability）

### 问题
- 用户不清楚为什么路由到对应 Agent，不信任系统
- 已有 audit_event 日志但不暴露给用户

### 方案

#### 2.1 决策事件构建器（新文件 `api/decision_explainer.py`）

从 ClassificationProcessor 的决策路径提取关键信息，构建用户可见的决策说明：

| 决策路径 | explanation 文本 |
|---------|-----------------|
| L1 switch-word | "检测到话题切换关键词，重新分类意图" |
| L2 low-confidence clarify | "输入涉及多个意图，需要澄清" |
| L3 context lock hit | "延续上轮对话，复用已有意图：{intent}" |
| L3 semantic continuation | "语义相似度 {score}，判定为延续对话" |
| Full classify + route | "意图分类：{intent}（置信度 {confidence}），路由到 {agent}" |

关键设计：
- 所有文本模板放 `prompts/decision_explain_v1.txt`
- 调用 `mask_sensitive()` 过滤敏感字段
- 不暴露内部 prompt/参数，只暴露意图+置信度+路由原因

#### 2.2 管道集成

修改 `classification_processor.py`：
- 在每个决策节点 yield `decision` 事件（SSE）
- decision 事件结构：`{intent_type, confidence, reason, agent, context_lock_status}`

#### 2.3 前端决策面板

修改 `web/templates/index.html`：
- 每条 AI 回复下方添加折叠面板"为什么这样回答？"
- 默认折叠，点击展开
- 展示内容：意图分类、置信度条、路由原因文字、Agent 名称
- 置信度 < 0.7 时用黄色标记

### 新问题应对：详情过多挤占页面 + 敏感信息泄露

| 问题 | 应对 |
|------|------|
| 详情文字过多挤占页面 | 折叠面板默认收起，只展示一句话摘要，展开才看详情 |
| 敏感信息泄露 | `mask_sensitive()` 过滤 + 白名单字段（只暴露 intent_type/confidence/reason/agent） |

### 新增/修改文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `api/decision_explainer.py` | 新建 | 决策说明构建器 |
| `prompts/decision_explain_v1.txt` | 新建 | 决策说明模板 |
| `agents/task_classification/classification_processor.py` | 修改 | yield decision 事件 |
| `web/templates/index.html` | 修改 | 折叠决策面板 |
| `tests/test_decision_explainer.py` | 新建 | 决策说明构建测试 |

---

## Feature 3: 任务可中断（feature/task-interruptible）

### 问题
- 任务启动后不能取消，终止后进度全部作废
- 无 asyncio.CancelledError 处理
- 无 task ID 追踪

### 方案

#### 3.1 任务管理器（新文件 `api/task_manager.py`）

```python
class TaskManager:
    """进程级任务追踪，管理 task_id → asyncio.Task 映射。"""
    _tasks: dict[str, asyncio.Task]  # task_id → asyncio.Task
    _checkpoints: dict[str, dict]     # task_id → checkpoint state

    def register(self, task_id, async_task) -> None
    def cancel(self, task_id) -> bool
    def checkpoint(self, task_id, state) -> None
    def get_checkpoint(self, task_id) -> dict | None
    def cleanup(self, task_id) -> None
```

设计要点：
- task_id = `{session_id}:{timestamp_hex}`（可追溯）
- asyncio.Task 存引用，cancel 调用 `task.cancel()`
- checkpoint 存最近中间结果（token 累积、处理阶段）
- TTL 300s 自动清理，防止内存泄漏

#### 3.2 Cancel 端点

修改 `web/routes.py`：
- 新增 `POST /chat/cancel`：body `{task_id}` 或 `{session_id}`
- 调用 `TaskManager.cancel(task_id)`
- 返回 `{cancelled: true, task_id, checkpoint_available: bool}`

#### 3.3 CancelledError 处理

修改 `agents/task_classification_agent.py::classify_task_stream`：
- 外层 `try/except asyncio.CancelledError`
- on cancel：save checkpoint to `_task_memory.checkpoint`，yield `done` 事件
- 不 re-raise（防止 500 错误）

修改 `classification_processor.py::process_task_stream`：
- 关键节点检查 `asyncio.current_task().cancelling()`（Python 3.11+）
- 如果被取消：保存当前状态（已分类的 intent、已收集的字段）

#### 3.4 断点续传

修改 `web/routes.py`：
- 新增 `POST /chat/resume`：body `{session_id}`
- 从 `_task_memory.checkpoint` 加载状态
- 从断点继续执行（如已分类但未路由 → 直接路由）

#### 3.5 前端取消按钮

修改 `web/templates/index.html`：
- 发送消息后显示"取消"按钮
- 点击 → fetch `/chat/cancel` → 显示"已取消"
- 如果有 checkpoint → 显示"继续"按钮

### 新问题应对：持久化中间结果存储开销 + 状态管理复杂

| 问题 | 应对 |
|------|------|
| 存储开销 | checkpoint 只存轻量元数据（intent_type, collected_fields, stage），不存全文 |
| 状态管理复杂 | TaskManager 单一入口，所有状态变更走 TaskManager，TTL 自动清理 |
| 竞态条件 | cancel 和正常完成可能竞态 → 用 asyncio.Lock 保护 task 状态变更 |

### 新增/修改文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `api/task_manager.py` | 新建 | 任务追踪 + 取消 + checkpoint |
| `web/routes.py` | 修改 | cancel + resume 端点 |
| `agents/task_classification_agent.py` | 修改 | CancelledError 处理 + checkpoint |
| `agents/task_classification/classification_processor.py` | 修改 | 阶段检查 + checkpoint 保存 |
| `api/chat_handler.py` | 修改 | task_id 生成 + TaskManager 注册 |
| `web/templates/index.html` | 修改 | 取消按钮 + 继续按钮 |
| `tests/test_task_manager.py` | 新建 | 任务取消 + checkpoint 测试 |

---

## Feature 4: 错误恢复（feature/error-recovery）

### 问题
- 失败后用户无解决办法，只能从头重新执行
- 错误信息不友好（当前只 yield `[error] ErrorTypeName`）
- 无重试机制

### 方案

#### 4.1 错误分类器（新文件 `api/error_classifier.py`）

```python
class ErrorCategory(Enum):
    RETRYABLE = "retryable"       # 超时、限流、网络波动
    NON_RETRYABLE = "non_retryable"  # 输入无效、认证失败

class ErrorInfo:
    category: ErrorCategory
    error_type: str               # TimeoutError, RateLimitError, etc.
    user_message: str             # 用户可见错误描述
    suggestion: str               # 建议操作
    retry_count: int = 0
    max_retries: int = 3
```

错误分类规则表：

| 异常类型 | 分类 | 用户消息 | 建议 |
|---------|------|---------|------|
| TimeoutError | retryable | "服务器响应超时" | "建议重试，如持续超时请检查网络" |
| RateLimitError | retryable | "请求过于频繁" | "请稍后重试" |
| ConnectionError | retryable | "网络连接异常" | "建议检查网络后重试" |
| ValueError/ValidationError | non_retryable | "输入格式有误" | "请修改输入后重新发送" |
| AuthError | non_retryable | "认证失败" | "请检查 API Key" |
| Unknown | non_retryable | "处理异常" | "请重新描述问题" |

#### 4.2 重试端点

修改 `web/routes.py`：
- 新增 `POST /chat/retry`：body `{session_id}`
- 从 session 历史取上一条用户消息
- 重走 `process_user_input_stream`，携带 `retry_count`
- `retry_count` 通过 ContextVar 传递，超 3 次拒绝

#### 4.3 错误事件集成

修改 `web/routes.py::token_generator`：
- catch Exception → 调用 `ErrorClassifier.classify(e)`
- yield SSE `error` 事件（含 category, suggestion, retryable）
- 非 retryable → 不显示重试按钮

#### 4.4 前端错误 UI

修改 `web/templates/index.html`：
- error 事件 → 显示错误卡片（红色背景）
- 卡片内容：错误描述 + 建议
- retryable=true → 显示"重试"按钮
- 点击重试 → fetch `/chat/retry` → 重新开始流式接收

### 新问题应对：区分可重试错误 + 无效重试增加服务器负载

| 问题 | 应对 |
|------|------|
| 不好区分可重试错误 | ErrorClassifier 基于异常类型 + 消息模式匹配，保守分类（unknown 默认 non_retryable） |
| 无效重试加大服务器负载 | max_retries=3 硬限，session 级计数，超限返回 429 |
| 同一错误反复重试 | 重试前检查上次错误类型，相同类型连续 2 次失败 → 建议联系管理员 |

### 新增/修改文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `api/error_classifier.py` | 新建 | 错误分类 + 建议引擎 |
| `prompts/error_suggestion_v1.txt` | 新建 | 错误建议模板（可选 LLM 生成建议） |
| `web/routes.py` | 修改 | retry 端点 + 结构化 error 事件 |
| `api/chat_handler.py` | 修改 | retry_count 传递 |
| `web/templates/index.html` | 修改 | 错误卡片 + 重试按钮 |
| `tests/test_error_classifier.py` | 新建 | 错误分类测试 |

---

## 实施顺序与依赖

```
Feature 1: SSE 流式协议       ← 基础设施，其他三个都依赖 SSE 事件
    ↓
Feature 2: 决策可解释性       ← 依赖 SSE 的 decision 事件
    ↓ (可并行)
Feature 3: 任务可中断         ← 依赖 SSE 的 error/done 事件
    ↓
Feature 4: 错误恢复           ← 依赖 SSE 的 error 事件 + task_id
```

实际执行：Feature 1 先做，合并后 Feature 2/3 可并行开发，Feature 4 最后。

---

## 约束遵守

- 单文件 ≤ 200 行，函数 ≤ 50 行
- 可替换组件走 `config/*_factory.py`（本次无新组件）
- 提示词放 `prompts/*.txt`
- MCP 错误用 `mcp_server/errors.py` 的 MCPError
- 输入校验在 `web/routes.py::ChatRequest` 加 field_validator
- 不在 agent/handler 里 `f"Error: {e}"` 暴露原始异常
- 测试不依赖真实 LLM（用 skipif + _has_api_key）
- 每 commit 完成一件小事，全绿才 commit
