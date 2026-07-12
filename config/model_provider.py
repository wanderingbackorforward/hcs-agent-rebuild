"""Model provider factory for chat LLMs and embeddings.

Supports Azure OpenAI and OpenAI-compatible providers such as Qwen,
DeepSeek, Zhipu, MiniMax and OpenAI by switching environment variables.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, List

import requests
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
    ChatOpenAI,
    OpenAIEmbeddings,
)
from pydantic import BaseModel, Field, SecretStr

load_dotenv()

CHAT_PROVIDERS = {"openai", "qwen", "deepseek", "zhipu", "minimax", "openai-compatible"}
EMBEDDING_PROVIDERS = {"openai", "qwen", "deepseek", "zhipu", "minimax", "openai-compatible"}
# local provider 支持内网私有化部署的 BGE / GTE 等模型，通过 sentence-transformers 本地推理
LOCAL_EMBEDDING_PROVIDERS = {"local"}


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def get_model_provider() -> str:
    return (_env("MODEL_PROVIDER", "openai") or "openai").strip().lower()


class MiniMaxEmbeddings(BaseModel, Embeddings):
    """MiniMax embedding client compatible with LangChain.

    Uses MiniMax's native ``/v1/embeddings`` endpoint which returns a top-level
    ``vectors`` array rather than the OpenAI-compatible shape.
    """

    model: str = "embo-01"
    api_key: SecretStr
    base_url: str = "https://api.minimaxi.com/v1/embeddings"
    timeout: int = Field(default=60, ge=1)

    def _embed(self, texts: List[str], embed_type: str) -> List[List[float]]:
        payload: dict[str, Any] = {
            "model": self.model,
            "type": embed_type,
            "texts": list(texts),
        }
        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            self.base_url, headers=headers, json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        base_resp = data.get("base_resp", {})
        if base_resp.get("status_code") != 0:
            raise ValueError(f"MiniMax embedding error: {base_resp}")
        return data["vectors"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts, "db")

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text], "query")[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed_query, text)


class LocalEmbeddings(BaseModel, Embeddings):
    """Local embedding model client for on-premise deployment.

    Supports BGE / GTE / m3e 等通过 sentence-transformers 加载的本地模型，
    适用于内网私有化部署场景（API 不可达或数据不出网）。

    模型通过 ``EMBEDDING_MODEL`` 指定，可以是 HuggingFace Hub 模型名
    （如 ``BAAI/bge-large-zh-v1.5``、``BAAI/bge-m3``）或本地路径。

    依赖 ``sentence-transformers`` 和 ``langchain-huggingface`` 包，
    未安装时在调用阶段抛出可读的安装指引，不影响其他 provider 的导入。
    """

    model: str = "BAAI/bge-large-zh-v1.5"
    device: str = "cpu"
    normalize_embeddings: bool = True
    _backend: Any = None

    def _get_backend(self):
        if self._backend is not None:
            return self._backend
        try:
            from langchain_huggingface import HuggingFaceEmbeddings as _Backend
        except ImportError:
            try:
                from langchain_community.embeddings import (
                    HuggingFaceEmbeddings as _Backend,
                )
            except ImportError:
                raise ImportError(
                    "Local embedding provider requires 'sentence-transformers' "
                    "and 'langchain-huggingface' (or 'langchain-community'). "
                    "Install with: pip install sentence-transformers langchain-huggingface"
                )
        self._backend = _Backend(
            model_name=self.model,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": self.normalize_embeddings},
        )
        return self._backend

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._get_backend().embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._get_backend().embed_query(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed_query, text)


def create_chat_model(temperature: float = 0):
    provider = get_model_provider()

    if provider == "azure":
        return AzureChatOpenAI(
            azure_deployment=_env("AZURE_OPENAI_DEPLOYMENT"),
            api_version=_env("AZURE_OPENAI_VERSION"),
            temperature=temperature,
            azure_endpoint=_env("AZURE_OPENAI_ENDPOINT"),
            api_key=SecretStr(_env("AZURE_OPENAI_API_KEY", "") or ""),
        )

    if provider in CHAT_PROVIDERS:
        return ChatOpenAI(
            model=_env("LLM_MODEL", "qwen-plus") or "qwen-plus",
            api_key=SecretStr(_env("LLM_API_KEY", "") or ""),
            base_url=_env("LLM_BASE_URL"),
            temperature=temperature,
        )

    raise ValueError(
        f"Unsupported MODEL_PROVIDER={provider!r}. "
        "Use azure, qwen, deepseek, zhipu, minimax, openai, or openai-compatible."
    )


def create_embedding_model():
    provider = (_env("EMBEDDING_PROVIDER") or get_model_provider()).strip().lower()

    if provider == "azure":
        return AzureOpenAIEmbeddings(
            azure_deployment=_env("AZURE_OPENAI_DEPLOYMENT_EMBEDDING"),
            api_key=SecretStr(_env("AZURE_OPENAI_API_KEY", "") or ""),
            api_version=_env("AZURE_OPENAI_EMBEDDING_VERSION", "2023-05-15"),
            azure_endpoint=_env("AZURE_OPENAI_ENDPOINT_EMBEDDING"),
        )

    if provider == "minimax":
        return MiniMaxEmbeddings(
            model=_env("EMBEDDING_MODEL", "embo-01") or "embo-01",
            api_key=SecretStr(_env("EMBEDDING_API_KEY") or _env("LLM_API_KEY", "") or ""),
            base_url=_env("EMBEDDING_BASE_URL", "https://api.minimaxi.com/v1/embeddings")
            or "https://api.minimaxi.com/v1/embeddings",
        )

    if provider in EMBEDDING_PROVIDERS:
        return OpenAIEmbeddings(
            model=_env("EMBEDDING_MODEL", "text-embedding-v3") or "text-embedding-v3",
            api_key=SecretStr(_env("EMBEDDING_API_KEY") or _env("LLM_API_KEY", "") or ""),
            base_url=_env("EMBEDDING_BASE_URL") or _env("LLM_BASE_URL"),
            check_embedding_ctx_length=False,
        )

    if provider in LOCAL_EMBEDDING_PROVIDERS:
        return LocalEmbeddings(
            model=_env("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
            or "BAAI/bge-large-zh-v1.5",
            device=_env("EMBEDDING_DEVICE", "cpu") or "cpu",
        )

    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER={provider!r}. "
        "Use azure, qwen, deepseek, zhipu, minimax, local, openai, or openai-compatible."
    )
