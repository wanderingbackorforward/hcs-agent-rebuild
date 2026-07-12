# ci: always upload pytest log as build artifact for auth-free inspec...

**日期**: 2026-06-14  
**提交数**: 8  

---

## 提交记录

| # | Commit | 说明 |
|---|--------|------|
| 1 | `5c18ec5` | ci: always upload pytest log as build artifact for auth-free inspection |
| 2 | `94d3972` | ci: re-enable python 3.10 + 3.11 in test matrix (was limited to 3.12) |
| 3 | `9cb3cc3` | ci: add real Linux deployment smoke (docker build + uvicorn + MCP stdio) |
| 4 | `ec1eb1c` | ci: quote step name containing colon (YAML parse error) |
| 5 | `4b768d7` | ci: quote second step name containing colon (YAML parse error) |
| 6 | `b530800` | ci: capture docker logs as artifact on smoke failure (auth-free debug) |
| 7 | `93c4409` | ci: upload docker-smoke.log as artifact |
| 8 | `ad4a588` | ci: dump docker logs inline in step output (no artifact download needed) |

## 变更文件

- `.github/workflows/tests.yml` (修改)

---

## 备注

<!-- 手动补充：升级背景、设计决策、面试话术等 -->
