# 项目 Redis 使用现状与引入建议

## 现状：完全未使用 Redis

- 项目当前没有使用 Redis
- requirements.txt 中无 redis 依赖
- 源码中无任何 import redis 或 Redis 客户端调用
- docker-compose.yml 中无 Redis 服务
- 项目中出现的 "redis" 字样仅两类：rate_limit.py 中的注释建议、种子数据/测试用例中的环境组件名称

## 当前存储与缓存方案

- 结构化数据：SQLite（SQLAlchemy ORM，4 张表：environments / validation_records / knowledge_documents / user_sessions）
- 向量数据：ChromaDB（PersistentClient，集合名 hcs_knowledge）
- 缓存：无，knowledge_qa_agent.py 中明确标注 "cache is a no-op"
- 限流：进程内 deque 滑动窗口（仅单进程有效）
- Agent 实例复用：chat_handler.py 中 _session_agents 字典（进程内对象缓存，非数据缓存）

## Redis 引入建议

### 1. 分布式限流（优先级：高）

- 当前 SlidingWindowLimiter 基于进程内 deque，多 worker/多实例部署时计数器不共享
- rate_limit.py 第 6 行已预留扩展注释："In a multi-process deployment, swap this for a Redis-backed limiter."
- 可用 Redis SLIDING WINDOW 或 TOKEN BUCKET 原子指令替换

### 2. LLM 调用结果缓存（优先级：高）

- 当前知识问答 Agent 每次调用 LLM 生成回答，语义相似的重复问题会重复消耗 token
- 可缓存 query_hash → LLM_response，设置 TTL 10-30 分钟
- 显著降低 LLM API 调用成本和响应延迟，投入产出比最高

### 3. 会话状态缓存（优先级：中）

- 当前 user_sessions 表存在 SQLite 中，每次请求查库
- 高频访问的 session 数据可缓存到 Redis，减少 DB 压力
- 多实例部署时可作 session 共享层

### 4. RAG 检索结果缓存（优先级：中）

- Hybrid Search（Dense + BM25 + RRF）计算开销较高
- 可缓存 query_hash → retrieved_doc_ids，避免重复检索

### 5. Agent 实例映射缓存（优先级：低）

- 当前 _session_agents 字典在进程内存中，重启即丢失
- Redis 可做轻量级 session-to-agent 映射缓存

## 推荐接入方式

- 推荐依赖：redis>=5.0（异步支持已内置，无需单独安装 aioredis）
- 连接方式：redis.asyncio as redis
- 默认地址：redis://localhost:6379/0（通过 REDIS_URL 环境变量配置）
- 项目当前为 MVP 阶段，SQLite + ChromaDB 够用
- 近期不计划多实例部署时，优先实施 LLM 结果缓存即可
