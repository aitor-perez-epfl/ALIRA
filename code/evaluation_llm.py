from typing import List
from abc import ABC, abstractmethod
from openai import OpenAI
import re


class AbstractEvaluationLLM(ABC):
    """Abstract base class for evaluation LLM services."""
    
    @abstractmethod
    def evaluate(self, texts: List[str], topic: str, evaluation_query: str = None) -> List[bool]:
        """Evaluate if texts are related to the topic."""
        raise NotImplementedError


class OpenAIEvaluationLLM(AbstractEvaluationLLM):
    """OpenAI evaluation LLM service implementation."""
    
    def __init__(self, api_key: str, model: str, max_attempts: int = 5):
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=api_key)
        self.max_attempts = max_attempts
        
    def evaluate(self, texts: List[str], topic: str, evaluation_query: str = None) -> List[bool]:
        """Evaluate texts using OpenAI API."""
        if evaluation_query is None:
            evaluation_query = "Classify each paper as related (1) or not related (0) to: {topic}"
        
        prompt = evaluation_query.format(topic=topic) + "\n\n"
        prompt += "\n".join(f"{i}. {t}" for i, t in enumerate(texts))
        prompt += f"\n\nReply with ONLY {len(texts)} digits (0 or 1), no spaces:"
        
        for attempt in range(self.max_attempts):
            try:
                try:
                    # Try using responses API (if available)
                    response = self.client.responses.create(
                        model=self.model,
                        input=prompt,
                        text={"format": {"type": "text"}, "verbosity": "low"},
                        reasoning={"effort": "high"}
                    )
                    output_text = response.output_text
                except AttributeError:
                    # Fallback to standard chat completions API
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    output_text = response.choices[0].message.content
                
                raw = re.sub(r'\s', '', output_text.strip())
                
                if len(raw) != len(texts) or not all(c in '01' for c in raw):
                    raise ValueError(f"Expected {len(texts)} binary digits, got: {raw[:100]}")
                
                related_flags = [c == '1' for c in raw]
                return related_flags
                
            except Exception as e:
                if attempt == self.max_attempts - 1:
                    print(f'Error (attempt {attempt + 1}/{self.max_attempts}): {e}')
                    return None
                print(f'Error (attempt {attempt + 1}/{self.max_attempts}): {e}')
        
        return None  # All retries failed


def create_evaluation_llm(api_key: str, model: str) -> AbstractEvaluationLLM:
    """
    Factory function to create evaluation LLM service from model name.
    
    Args:
        api_key: API key for the service
        model: Model name (e.g., "gpt-5.2")
        
    Returns:
        Evaluation LLM service instance
        
    Raises:
        ValueError if model not recognized
    """
    if model == "gpt-5.2":
        return OpenAIEvaluationLLM(api_key=api_key, model=model)
    elif model == "gpt-4":
        return OpenAIEvaluationLLM(api_key=api_key, model=model)
    elif model == "gpt-4-turbo":
        return OpenAIEvaluationLLM(api_key=api_key, model=model)
    else:
        raise ValueError(f"Unknown evaluation LLM model: {model}")
