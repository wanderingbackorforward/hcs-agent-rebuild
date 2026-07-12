# Skills

项目辅助脚本集合。每个 skill 一个子文件夹，独立运作。

| Skill | 说明 | 入口 |
|-------|------|------|
| `git-trace` | 按「升级」分组追溯 git commit，生成 Markdown 追溯文档 | `git-trace/trace_upgrades.py` |

## 新增 skill

1. 在本目录下新建子文件夹（如 `my-skill/`）
2. 脚本 + 说明放同一文件夹
3. 在上方表格追加一行
4. `git add skills/ && git commit`
