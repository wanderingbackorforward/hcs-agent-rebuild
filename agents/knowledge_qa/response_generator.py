"""Response generator - generates natural language answer from retrieved chunks."""
import logging

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


class ResponseGenerator:
    def __init__(self, llm):
        self.llm = llm

    async def generate(self, query: str, results: list) -> str:
        if not results:
            return "未在知识库中找到相关资料。请尝试换一种说法，或联系管理员补充文档。"

        context = ""
        for i, (doc_id, text, score, meta) in enumerate(results[:5], 1):
            title = meta.get("title", doc_id)
            context += f"[{i}] 来源：{title}\n{text}\n\n"

        prompt = f"""你是 HCS 测试辅助助手。请严格根据以下知识库内容回答用户问题。
如果资料不足以回答，请明确说明。

## 知识库内容
{context}

## 用户问题
{query}

## 答案（简洁、准确，使用中文）："""

        try:
            full = ""
            async for chunk in self.llm.astream([HumanMessage(content=prompt)]):
                full += chunk.content
            return full.strip()
        except Exception as e:
            logger.warning(f"LLM answer generation failed: {e}")
            # Fallback: return top result snippets
            return "根据知识库检索结果：\n" + "\n".join(
                f"- {meta.get('title', doc_id)}: {text[:200]}..."
                for doc_id, text, score, meta in results[:3]
            )
