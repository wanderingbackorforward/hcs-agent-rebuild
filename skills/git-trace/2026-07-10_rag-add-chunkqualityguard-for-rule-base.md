# rag: add ChunkQualityGuard for rule-based chunk pre-filtering

**日期**: 2026-07-10  
**提交数**: 1  

---

## 提交记录

| # | Commit | 说明 |
|---|--------|------|
| 1 | `07a9a63` | feat(rag): add ChunkQualityGuard for rule-based chunk pre-filtering |

## 变更文件

- `.env.example` (修改)
- `config/chunker_factory.py` (修改)
- `config/settings.py` (修改)
- `rag/ingestion/chunking/quality_guard.py` (新增)
- `rag/ingestion/pipeline.py` (修改)
- `tests/test_chunk_quality_guard.py` (新增)

---

## 备注

<!-- 手动补充：升级背景、设计决策、面试话术等 -->
