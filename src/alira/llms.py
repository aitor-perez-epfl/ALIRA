import logging

from openai import OpenAI

from alira.config import CONFIG

logging.getLogger("httpx").setLevel(logging.WARNING)


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
