# git-trace — Git 升级追溯工具

按「升级」分组追溯 git commit，每个升级组生成一份 Markdown 文档，统一存放在本文件夹。

## 快速使用

```bash
# 从项目根目录运行
python skills/git-trace/trace_upgrades.py --output skills/git-trace

# 只看分组预览，不写文件
python skills/git-trace/trace_upgrades.py --dry-run

# 重新生成全部（覆盖已有文件）
python skills/git-trace/trace_upgrades.py --force --output skills/git-trace
```

## 分组策略

| 策略 | 触发条件 | 示例 |
|------|---------|------|
| merge 分组 | 遇到 `merge: feature/xxx` 提交 | U8 SSE 的 3 个 feat + 1 个 merge → 一份文档 |
| scope 分组 | 连续提交的 conventional commit scope 相同 | 6 个 `feat(memory):` → 一份文档 |
| 日期拆分 | 同 scope 但日期间隔 >2 天 | 拆为两个组 |

## 生成的文件

每个升级组一个 `.md` 文件，命名格式：

- merge 组：`{日期}_U{编号}-{feature}.md`（如 `2026-07-09_U8-sse-streaming.md`）
- scope 组：`{日期}_{scope}-{描述摘要}.md`（如 `2026-07-08_memory-add-layered-memory-skeleton.md`）

每份文档包含：

- 标题 + 日期 + 提交数 + 分支（merge 组）
- 提交记录表（hash + message）
- 变更文件列表（路径 + 操作类型）
- 备注区（手动补充：升级背景、设计决策、面试话术等）

`INDEX.md` 是自动生成的总索引表。

## 增量更新

新增 commit 后重新运行脚本，已存在的文件自动跳过（除非 `--force`）。
`INDEX.md` 每次运行都会刷新。

## 文件清单

```
skills/git-trace/
├── trace_upgrades.py          # 脚本
├── README.md                  # 本文件
├── INDEX.md                   # 自动生成的索引
├── 2026-06-14_*.md            # 追溯文档（按日期排序）
├── 2026-07-08_*.md
├── 2026-07-09_U8-sse-streaming.md
└── ...
```
