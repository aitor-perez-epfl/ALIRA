from pydantic import BaseModel, ValidationError

from openai import OpenAI, APITimeoutError

from alira.config import CONFIG


def send_embedding_request(texts: list[str]) -> list[list[float]]:
    client = OpenAI(base_url=CONFIG['ALIRA_LLM_BASE_URL'], api_key=CONFIG['ALIRA_LLM_API_KEY'])
    embeddings = []
    batch_size = 500

    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    for batch in batches:
        r = client.embeddings.create(model=CONFIG['ALIRA_LLM_EMBEDDING_MODEL'], input=batch)
        embeddings += [item.embedding for item in r.data]

    return embeddings


def send_llm_request(messages, response_format=None, max_tokens=None, timeout=30):
    if response_format:
        response_format_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": response_format.__name__,
                "schema": response_format.model_json_schema(),
                "strict": True,
            },
        }
    else:
        response_format_schema = None

    client = OpenAI(base_url=CONFIG['ALIRA_LLM_BASE_URL'], api_key=CONFIG['ALIRA_LLM_API_KEY'], timeout=timeout)
    response = client.chat.completions.create(
        model=CONFIG['ALIRA_LLM_BASE_MODEL'],
        messages=messages,
        response_format=response_format_schema,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content.strip()

    if response_format:
        return response_format.model_validate_json(content)

    return content


def generate_documents(
    query: str,
    n: int,
    examples: list[str],
    prompt: str | None = None,
    max_retries: int = 3,
) -> list[str]:
    """Generate n synthetic documents about query, using examples for format reference."""

    if prompt is None:
        examples_block = "\n\n---\n\n".join(
            f"{example}"
            for i, example in enumerate(examples)
        )
        prompt = f"""
You are an expert generating texts related to a given topic. 
Below are some randomly chosen example texts from a dataset.
Your task is to produce a list of exactly {n} documents with the same format as the example texts, but so that all the texts you produce are related to the topic "{query}".
Produce texts that could be extracted from the same dataset as the examples but somehow having filtered semantically by the given topic.
Do not include the delimiters in your generated texts.

Here are the examples:

---

{examples_block}

---
"""

    messages = [{'role': 'user', 'content': prompt}]

    class TextList(BaseModel):
        texts: list[str]

    for attempt in range(max_retries):
        try:
            response = send_llm_request(messages, response_format=TextList)
            if len(response.texts) == n:
                return response.texts
        except APITimeoutError as e:
            print(f"Failed to generate texts before timeout. Error: {e}")
            pass
        except ValidationError as e:
            print(f"Failed to generate exactly {n} texts. Error: {e}")
            pass

    raise ValueError(f"Failed to get exactly {n} texts after {max_retries} attempts")


def evaluate_documents(
    query: str,
    texts: list[str],
    prompt: str | None = None,
    max_retries: int = 3,
) -> list[bool]:
    """Evaluate with an LLM whether each of the texts is related to query."""

    n = len(texts)

    if n == 0:
        return []

    if prompt is None:
        numbered = "\n".join(f"{i}. {text}" for i, text in enumerate(texts))
        prompt = f"""
You are an expert in classifying documents according to a given topic.
Classify each document as *related* (True) or *not related* (False) with the topic "{query}".
Produce a list of exactly {n} bools, one for each document, in the same order as the documents.

{numbered}
"""

    messages = [{'role': 'user', 'content': prompt}]
    max_tokens = n * 8 + 32

    class EvaluationList(BaseModel):
        evaluations: list[bool]

    for attempt in range(max_retries):
        try:
            response = send_llm_request(messages, response_format=EvaluationList, max_tokens=max_tokens)
            if len(response.evaluations) == n:
                return response.evaluations
        except APITimeoutError as e:
            print(f"Failed to generate evaluations before timeout. Error: {e}")
            pass
        except ValidationError as e:
            print(f"Failed to generate exactly {n} evaluations. Error: {e}")
            pass

    raise ValueError(f"Failed to get exactly {n} evaluations after {max_retries} attempts")
