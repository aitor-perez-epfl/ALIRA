from typing import List
from abc import ABC, abstractmethod
from openai import OpenAI
import time


class AbstractGenerationLLM(ABC):
    """Abstract base class for generation LLM services."""
    
    @abstractmethod
    def generate_titles(self, topic: str, n_titles: int) -> List[str]:
        """Generate synthetic titles for a topic."""
        raise NotImplementedError


class OpenAIGenerationLLM(AbstractGenerationLLM):
    """OpenAI generation LLM service implementation."""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=api_key)
        
    def generate_titles(self, topic: str, n_titles: int) -> List[str]:
        """Generate synthetic titles using OpenAI API."""
        prompt = f"""
You are an expert academic writer. 
Produce exactly {n_titles} titles of academic writing followed by a very short abstracts (1 sentence) about the following topic:
{topic}

Separate each title + abstract by a new line. So the final format look like this:
This is an Example Title. This is the start of an abstract
This is the Second Example Title. This is the second start of the abstract
"""
        try:
            # Try using responses API (if available)
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
            )
            output_text = response.output_text
        except AttributeError:
            # Fallback to standard chat completions API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            output_text = response.choices[0].message.content
        
        titles = [line.strip() for line in output_text.split("\n") if line.strip()]
        
        if len(titles) < n_titles - 1:
            return None
            
        return titles


def create_generation_llm(api_key: str, model: str) -> AbstractGenerationLLM:
    """
    Factory function to create generation LLM service from model name.
    
    Args:
        api_key: API key for the service
        model: Model name (e.g., "gpt-4o-mini")
        
    Returns:
        Generation LLM service instance
        
    Raises:
        ValueError if model not recognized
    """
    if model == "gpt-4o-mini":
        return OpenAIGenerationLLM(api_key=api_key, model=model)
    elif model == "gpt-3.5-turbo":
        return OpenAIGenerationLLM(api_key=api_key, model=model)
    elif model == "gpt-4":
        return OpenAIGenerationLLM(api_key=api_key, model=model)
    elif model == "gpt-5-mini":
        return OpenAIGenerationLLM(api_key=api_key, model=model)
    else:
        raise ValueError(f"Unknown generation LLM model: {model}")
