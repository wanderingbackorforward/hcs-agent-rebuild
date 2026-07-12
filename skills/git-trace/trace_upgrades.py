#!/usr/bin/env python3
"""
Git 升级追溯工具
================
按「升级」分组追溯 git commit，每个升级组生成一份 Markdown 文档。

分组策略：
  1. merge 提交（merge: feature/xxx）触发分组刷新 —— 其前的同 scope 提交 + merge 本身归为一组
  2. 其余提交按 conventional commit 的 scope（无 scope 则按 type）连续分组
  3. 同 key 但日期间隔 >2 天则拆分为不同组

用法：
  python trace_upgrades.py              # 增量生成（跳过已存在的文件）
  python trace_upgrades.py --force      # 重新生成全部
  python trace_upgrades.py --dry-run    # 只预览不写文件
  python trace_upgrades.py --output .   # 指定输出目录（默认当前目录）
"""

import argparse
import os
import re
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path


# ── git 操作 ──────────────────────────────────────────────

def run_git(args, cwd=None):
    result = subprocess.run(
        ["git"] + args,
        capture_output=True,
        text=True,
        cwd=cwd or os.getcwd(),
        check=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def get_commits():
    """获取当前分支全部提交（时间正序，oldest first）。"""
    output = run_git(
        ["log", "--pretty=format:%H|%an|%ad|%s", "--date=short", "--reverse"]
    )
    commits = []
    for line in output.split("\n"):
        if not line.strip():
            continue
        parts = line.split("|", 3)
        if len(parts) < 4:
            continue
        hash_, author, date, subject = parts
        commits.append(
            {
                "hash": hash_,
                "short_hash": hash_[:7],
                "author": author,
                "date": date,
                "subject": subject,
            }
        )
    return commits


def get_files_changed(commit_hash):
    """获取单个提交的变更文件列表。"""
    try:
        output = run_git(
            ["diff-tree", "--no-commit-id", "--name-status", "-r", commit_hash]
        )
    except subprocess.CalledProcessError:
        return []
    files = []
    for line in output.split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            files.append({"status": parts[0], "path": parts[-1]})
    return files


# ── conventional commit 解析 ─────────────────────────────

COMMIT_PATTERN = re.compile(
    r"^(feat|fix|docs|refactor|merge|test|chore|style|perf|build|ci)"
    r"(?:\(([^)]+)\))?"
    r"\s*:\s*(.*)",
    re.IGNORECASE,
)


def parse_commit(subject):
    """返回 (type, scope, description)，不匹配则 (None, None, subject)。"""
    m = COMMIT_PATTERN.match(subject)
    if m:
        return m.group(1).lower(), m.group(2), m.group(3)
    return None, None, subject


# ── 分组逻辑 ──────────────────────────────────────────────

MERGE_PATTERN = re.compile(
    r"merge:\s*feature/(\S+)\s*[—\-–]\s*(.*)", re.IGNORECASE
)
U_NUM_PATTERN = re.compile(r"\(U(\d+)\)")


def group_commits(commits):
    """
    按升级分组。

    返回 list[dict]，每个 dict:
      title       — 组标题
      commits     — 提交列表
      is_merge    — 是否由 merge 提交收尾
      u_number    — U 编号（如有）
      feature     — feature 分支名（如有）
    """
    groups = []
    buf = []
    buf_key = None
    buf_date = None

    for commit in commits:
        ctype, scope, desc = parse_commit(commit["subject"])

        # ── merge 提交：刷出缓冲区 + merge 本身为一个完整组 ──
        if ctype == "merge":
            buf.append(commit)
            m = MERGE_PATTERN.match(commit["subject"])
            feature = m.group(1) if m else "unknown"
            raw_desc = m.group(2) if m else commit["subject"]
            u_match = U_NUM_PATTERN.search(commit["subject"])
            u_num = u_match.group(1) if u_match else None

            title = raw_desc.strip()
            if u_num:
                title = f"[U{u_num}] {title}"

            groups.append(
                {
                    "title": title,
                    "commits": list(buf),
                    "is_merge": True,
                    "u_number": u_num,
                    "feature": feature,
                }
            )
            buf = []
            buf_key = None
            buf_date = None
            continue

        # ── 普通提交：按 scope/type 分组 ──
        # 忽略 issue 编号 scope (#123) → 退化为 type
        if scope and scope.startswith("#"):
            key = ctype or "other"
        else:
            key = scope or ctype or "other"

        cur_date = datetime.strptime(commit["date"], "%Y-%m-%d")

        should_flush = False
        if buf and key != buf_key:
            should_flush = True
        elif buf_date and (cur_date - buf_date).days > 2:
            should_flush = True

        if should_flush:
            groups.append(_make_scope_group(buf))
            buf = []

        buf.append(commit)
        buf_key = key
        buf_date = cur_date

    if buf:
        groups.append(_make_scope_group(buf))

    return groups


def _make_scope_group(commits):
    first = commits[0]
    ctype, scope, desc = parse_commit(first["subject"])
    if scope:
        title = f"{scope}: {desc}" if desc else scope
    elif ctype:
        title = f"{ctype}: {desc}" if desc else ctype
    else:
        title = first["subject"]
    if len(title) > 70:
        title = title[:67] + "..."
    return {
        "title": title,
        "commits": list(commits),
        "is_merge": False,
        "u_number": None,
        "feature": None,
    }


# ── 文件名 & slug ─────────────────────────────────────────

def slugify(text):
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[\s_]+", "-", text)
    return text.strip("-").lower()


def generate_filename(group):
    first = group["commits"][0]
    date = first["date"]

    if group["is_merge"]:
        slug = slugify(group["feature"] or "upgrade")
        if group["u_number"]:
            return f"{date}_U{group['u_number']}-{slug}.md"
        return f"{date}_{slug}.md"

    ctype, scope, desc = parse_commit(first["subject"])
    key = scope or ctype or "change"
    if scope and scope.startswith("#"):
        key = ctype or "fix"
    desc_slug = slugify(desc)[:35] if desc else first["short_hash"]
    if not desc_slug:
        desc_slug = first["short_hash"]
    return f"{date}_{slugify(key)}-{desc_slug}.md"


# ── 文档生成 ──────────────────────────────────────────────

STATUS_LABEL = {"A": "新增", "M": "修改", "D": "删除", "R": "重命名", "C": "复制"}


def generate_trace(group):
    lines = []
    first = group["commits"][0]
    last = group["commits"][-1]

    lines.append(f"# {group['title']}")
    lines.append("")

    date_range = first["date"]
    if last["date"] != first["date"]:
        date_range = f"{first['date']} ~ {last['date']}"
    lines.append(f"**日期**: {date_range}  ")
    lines.append(f"**提交数**: {len(group['commits'])}  ")
    if group["is_merge"] and group["feature"]:
        lines.append(f"**分支**: `feature/{group['feature']}`  ")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 提交记录表
    lines.append("## 提交记录")
    lines.append("")
    lines.append("| # | Commit | 说明 |")
    lines.append("|---|--------|------|")
    for i, c in enumerate(group["commits"], 1):
        lines.append(f"| {i} | `{c['short_hash']}` | {c['subject']} |")
    lines.append("")

    # 变更文件
    all_files = OrderedDict()
    for c in group["commits"]:
        for f in get_files_changed(c["hash"]):
            label = STATUS_LABEL.get(f["status"][0], f["status"])
            if f["path"] not in all_files:
                all_files[f["path"]] = label

    if all_files:
        lines.append("## 变更文件")
        lines.append("")
        for path, label in all_files.items():
            lines.append(f"- `{path}` ({label})")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 备注")
    lines.append("")
    lines.append("<!-- 手动补充：升级背景、设计决策、面试话术等 -->")
    lines.append("")

    return "\n".join(lines)


def generate_index(groups):
    lines = []
    lines.append("# Git 升级追溯索引")
    lines.append("")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines.append(f"> 自动生成 by `trace_upgrades.py` | 更新于 {now}")
    lines.append("")
    lines.append(f"共 **{len(groups)}** 个升级组。")
    lines.append("")
    lines.append("| # | 文件 | 日期 | 提交数 | 标题 |")
    lines.append("|---|------|------|--------|------|")
    for i, g in enumerate(groups, 1):
        fname = generate_filename(g)
        date = g["commits"][0]["date"]
        n = len(g["commits"])
        lines.append(f"| {i} | [{fname}]({fname}) | {date} | {n} | {g['title']} |")
    lines.append("")
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Git 升级追溯工具")
    parser.add_argument("--force", action="store_true", help="重新生成全部文件")
    parser.add_argument("--dry-run", action="store_true", help="只预览不写文件")
    parser.add_argument(
        "--output", default=".", help="输出目录（默认: 脚本所在目录）"
    )
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # 检查 git 仓库
    try:
        run_git(["rev-parse", "--git-dir"])
    except subprocess.CalledProcessError:
        print("错误：当前目录不在 git 仓库中", file=sys.stderr)
        sys.exit(1)

    print("正在获取提交历史...")
    commits = get_commits()
    print(f"  共 {len(commits)} 个提交")

    print("正在分组...")
    groups = group_commits(commits)
    print(f"  共 {len(groups)} 个升级组:\n")
    for g in groups:
        tag = " [merge]" if g["is_merge"] else ""
        print(f"  • {g['title']}{tag} ({len(g['commits'])} commits)")

    if args.dry_run:
        print("\n[dry-run] 未写入任何文件")
        return

    print("\n正在生成追溯文件...")
    generated = 0
    skipped = 0
    for g in groups:
        fname = generate_filename(g)
        fpath = output_dir / fname
        if fpath.exists() and not args.force:
            print(f"  跳过: {fname}")
            skipped += 1
            continue
        content = generate_trace(g)
        fpath.write_text(content, encoding="utf-8")
        print(f"  生成: {fname}")
        generated += 1

    # 索引
    index_path = output_dir / "INDEX.md"
    index_path.write_text(generate_index(groups), encoding="utf-8")
    print(f"\n  索引: INDEX.md")

    print(f"\n完成: 生成 {generated} 个文件, 跳过 {skipped} 个已存在文件")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
