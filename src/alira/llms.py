from pydantic import BaseModel, Field

from openai import OpenAI

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

    client = OpenAI(base_url=CONFIG['ALIRA_LLM_BASE_URL'], api_key=CONFIG['ALIRA_LLM_API_KEY'])
    response = client.chat.completions.create(
        model=CONFIG['ALIRA_LLM_BASE_MODEL'],
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

    messages = [{'role': 'user', 'content': prompt}]

    class TextItem(BaseModel):
        text: str

    class TextList(BaseModel):
        texts: list[TextItem] = Field(min_length=n, max_length=n)

    text_list = send_llm_request(messages, response_format=TextList)

    return [text_item.text for text_item in text_list.texts]


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

    class EvaluationList(BaseModel):
        evaluations: list[bool] = Field(min_length=n, max_length=n)

    evaluation_list = send_llm_request(messages, response_format=EvaluationList)

    return evaluation_list.evaluations
