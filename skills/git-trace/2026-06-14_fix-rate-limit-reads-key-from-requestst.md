# fix: rate_limit reads key from request.state (was being treated as ...

**日期**: 2026-06-14  
**提交数**: 1  

---

## 提交记录

| # | Commit | 说明 |
|---|--------|------|
| 1 | `1e44ea6` | fix: rate_limit reads key from request.state (was being treated as query param) |

## 变更文件

- `api/auth.py` (修改)
- `api/rate_limit.py` (修改)

---

## 备注

<!-- 手动补充：升级背景、设计决策、面试话术等 -->
