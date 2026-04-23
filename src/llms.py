from typing import Annotated, List

from pydantic import BaseModel, Field, RootModel

from openai import OpenAI

from alira.config import config


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

    # Send request
    rcp_client = OpenAI(base_url=config['RCP_BASE_URL'], api_key=config['RCP_API_KEY'])
    response = rcp_client.chat.completions.create(model=config['RCP_BASE_MODEL'], messages=messages, response_format=response_format_schema)
    content = response.choices[0].message.content.strip()

    # Return parsed result if structured output
    if response_format:
        return response_format.model_validate_json(content)

    # Return string otherwise
    return content


def generate_documents(topic: str, n: int, document_type: str) -> list[str]:
    """Generate synthetic documents of the given type about the given topic using an LLM."""

    prompt = f"""
You are an expert in writing documents of the type `{document_type}` about a given topic.
Produce a list of exactly {n} documents of the type `{document_type}` related to the topic "{topic}".
Each should consist of a name or title and a brief representative piece of text (e.g. an abstract for a publication, a description for a grant, etc.).
"""

    class Document(BaseModel):
        name: str
        description: str

    messages = [{'role': 'user', 'content': prompt}]
    list_model = RootModel[Annotated[List[Document], Field(min_length=n, max_length=n)]]
    documents = send_llm_request(messages, response_format=list_model)

    documents = ['\n'.join([document.name, document.description]) for document in documents.root]

    return documents


def evaluate_documents(topic: str, texts: list) -> list:
    """Evaluate whether each of the documents is related to the given topic using an LLM."""

    n = len(texts)

    prompt = f"""
You are an expert in classifying documents according to a given topic.
Classify each document as *related* (True) or *not related* (False) with the topic "{topic}".
Produce a list of exactly {n} bools, one for each document, in the same order as the documents.

{"\n".join(f"{i}. {text}" for i, text in enumerate(texts))}
"""

    messages = [{'role': 'user', 'content': prompt}]
    list_model = RootModel[Annotated[List[bool], Field(min_length=n, max_length=n)]]
    evaluations = send_llm_request(messages, response_format=list_model)

    return evaluations.root


if __name__ == '__main__':
    topic = 'machine learning'
    n = 3

    publications = generate_documents(topic, n, 'publication')

    print(publications)

    evaluations = evaluate_documents(topic, publications)

    print(evaluations)
