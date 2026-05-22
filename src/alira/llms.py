from typing import Annotated, List

from pydantic import BaseModel, Field, RootModel

from openai import OpenAI

from alira.config import CONFIG


def send_embedding_request(texts: list[str]) -> list[list[float]]:
    client = OpenAI(base_url=CONFIG['RCP_BASE_URL'], api_key=CONFIG['RCP_API_KEY'])
    response = client.embeddings.create(model=CONFIG['RCP_EMBEDDING_MODEL'], input=texts)
    return [item.embedding for item in response.data]


def send_llm_request(messages, response_format=None):
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

    client = OpenAI(base_url=CONFIG['RCP_BASE_URL'], api_key=CONFIG['RCP_API_KEY'])
    response = client.chat.completions.create(
        model=CONFIG['RCP_BASE_MODEL'],
        messages=messages,
        response_format=response_format_schema,
    )
    content = response.choices[0].message.content.strip()

    if response_format:
        return response_format.model_validate_json(content)

    return content


def generate_documents(
    query: str,
    n: int,
    examples: list[str],
    prompt: str | None = None
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

    class Text(BaseModel):
        text: str

    messages = [{'role': 'user', 'content': prompt}]
    list_model = RootModel[Annotated[List[Text], Field(min_length=n, max_length=n)]]
    texts = send_llm_request(messages, response_format=list_model)

    return [text.text for text in texts.root]


def evaluate_documents(
    query: str,
    texts: list[str],
    prompt: str | None = None
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
    list_model = RootModel[Annotated[List[bool], Field(min_length=n, max_length=n)]]
    evaluations = send_llm_request(messages, response_format=list_model)

    return evaluations.root
