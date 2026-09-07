import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app.llm.providers import PROVIDERS
from app.llm.config import LLM_NODES, EMBEDDING_NODE

def get_chat_model(node: str) -> ChatOpenAI:
    cfg = LLM_NODES[node]
    provider = PROVIDERS[cfg["provider"]]
    return ChatOpenAI(
        base_url=provider["base_url"],
        api_key=os.environ[provider["api_key_env"]],
        model=cfg["model"],
    )

def _embeddings_client() -> OpenAIEmbeddings:
    provider = PROVIDERS[EMBEDDING_NODE["provider"]]
    return OpenAIEmbeddings(
        base_url=provider["base_url"],
        api_key=os.environ[provider["api_key_env"]],
        model=EMBEDDING_NODE["model"],
    )

def embed(text: str) -> list[float]:
    return _embeddings_client().embed_query(text)
