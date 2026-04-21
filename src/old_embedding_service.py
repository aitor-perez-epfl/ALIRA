from typing import List
from abc import ABC, abstractmethod
from openai import OpenAI
from more_itertools import chunked


class AbstractEmbeddingService(ABC):
    """Abstract base class for embedding services."""
    
    @abstractmethod
    def embed_texts(self, texts: List[str], batch_size: int = 256) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        raise NotImplementedError
        
    @abstractmethod
    def get_model_info(self) -> dict:
        """Return info about the embedding model (provider, model name)."""
        raise NotImplementedError


class OpenAIEmbeddingService(AbstractEmbeddingService):
    """OpenAI embedding service implementation."""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=api_key)
        
    def embed_texts(self, texts: List[str], batch_size: int = 256) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        embeddings = []
        for batch in chunked(texts, batch_size):
            resp = self.client.embeddings.create(
                input=batch,
                model=self.model,
            )
            embeddings.extend(item.embedding for item in resp.data)
        return embeddings
        
    def get_model_info(self) -> dict:
        """Return OpenAI embedding model info."""
        return {"provider": "openai", "model": self.model}


def create_embedding_service(api_key: str, model: str) -> AbstractEmbeddingService:
    """
    Factory function to create embedding service from model name.
    
    Args:
        api_key: API key for the service
        model: Model name (e.g., "text-embedding-3-small")
        
    Returns:
        Embedding service instance
        
    Raises:
        ValueError if model not recognized
    """
    if model == "text-embedding-3-small":
        return OpenAIEmbeddingService(api_key=api_key, model=model)
    elif model == "text-embedding-3-large":
        return OpenAIEmbeddingService(api_key=api_key, model=model)
    elif model == "text-embedding-ada-002":
        return OpenAIEmbeddingService(api_key=api_key, model=model)
    else:
        raise ValueError(f"Unknown embedding model: {model}")
