import asyncio

from pydantic import BaseModel

from alira.llms import send_llm_request


async def _evaluate_async(
    query: str,
    texts: list[str],
    prompt: str | None = None,
) -> list[bool]:
    """Async implementation: one LLM call per text, max 5 concurrent."""

    if not texts:
        return []

    semaphore = asyncio.Semaphore(5)

    class Evaluation(BaseModel):
        related: bool

    if prompt is None:
        prompt = f"""\
You are an expert in classifying texts according to a given topic.
Classify the following text as *related* (true) or *not related* (false) with the topic "{query}"."""

    async def _evaluate_one_async(text: str) -> bool:
        doc_prompt = f"{prompt}\n\n---\n\n{text}"
        messages = [{'role': 'user', 'content': doc_prompt}]

        async with semaphore:
            response = await asyncio.to_thread(
                send_llm_request,
                messages,
                response_format=Evaluation,
                max_tokens=16,
            )
            return response.related

    return await asyncio.gather(*[_evaluate_one_async(text) for text in texts])


def evaluate(
    query: str,
    texts: list[str],
    prompt: str | None = None,
) -> list[bool]:
    """Evaluate with an LLM whether each of the texts is related to query.

    Makes parallel LLM calls, one per text, with a maximum of 5 concurrent requests.
    """

    return asyncio.run(_evaluate_async(query, texts, prompt))
