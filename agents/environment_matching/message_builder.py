"""Message builder for environment matching multi-turn conversation."""


class EnvironmentMessageBuilder:
    FIELD_NAMES = {
        "env_type": "环境类型（如 dev/test/staging）",
        "components": "需要的组件（如 mysql、redis、kafka）",
        "region": "区域（如 beijing/shanghai）",
        "service_status": "服务状态（available/busy/unknown）",
        "resource_usage": "资源占用要求",
    }

    def build_question(self, missing_fields: list) -> str:
        if not missing_fields:
            return "环境条件已收集完整，正在为您匹配候选环境..."

        missing_labels = [self.FIELD_NAMES.get(f, f) for f in missing_fields]
        return (
            "为了帮您精准匹配 HCS 测试环境，还需要确认以下信息：\n"
            + "\n".join(f"- {label}" for label in missing_labels)
            + "\n\n请补充说明。"
        )

    def build_candidates(self, candidates: list) -> str:
        if not candidates:
            return "未找到符合当前条件的候选环境，请放宽条件或补充更多信息。"

        lines = [f"为您找到 {len(candidates)} 个候选环境："]
        for c in candidates:
            lines.append(
                f"- **{c['name']}** ({c['env_type']}, {c['region']})\n"
                f"  组件：{', '.join(c['components'] or [])}\n"
                f"  状态：{c['status']} | 地址：{c['host']}:{c['port']}"
            )
        return "\n".join(lines)

    def build_validation(self, validation: dict) -> str:
        if not validation.get("valid"):
            return (
                f"环境验证未通过：{validation.get('probe_result', {}).get('error', '未知错误')}。"
            )
        return (
            f"环境验证通过。已匹配组件：{', '.join(validation.get('matched_components', []))}。"
            f"探测结果：{validation.get('probe_result', {})}。"
        )
