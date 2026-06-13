"""Unrelated handler - politely redirect off-topic inputs."""


class UnrelatedHandler:
    def __init__(self, state_manager):
        self.state_manager = state_manager

    async def handle(self, user_input: str) -> str:
        self.state_manager.reset()
        return (
            "您好，我是 HCS 测试辅助助手，专注于帮助您：\n"
            "1. 确认和匹配 HCS 测试环境（环境类型、组件、服务状态等）\n"
            "2. 查询 SDK 文档、用户手册、内部测试规范等技术资料\n\n"
            "请问有什么可以帮您的？"
        )
