import re
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


def generate_documents(topic: str, n: int, model: BaseModel) -> list:
    """Generate synthetic documents using an LLM."""

    # Turn CamelCased class name into readable words
    document_type = re.sub(r'(?<!^)(?=[A-Z])', ' ', model.__name__).lower()
    document_description = model.__doc__

    prompt = f"""
You are an expert in writing {document_type} content about "{topic}".
An item of {document_type} is described as {document_description}.
Produce a list of exactly {n} items of {document_type} about "{topic}".
"""

    messages = [{'role': 'user', 'content': prompt}]
    list_model = RootModel[Annotated[List[model], Field(min_length=n, max_length=n)]]
    model_instances = send_llm_request(messages, response_format=list_model)

    return model_instances.root


def evaluate_documents(topic: str, texts: list) -> list:
    """Evaluate whether each of the documents is related to the given topic using an LLM."""

    n = len(texts)

    prompt = f"""
You are an expert in classifying documents about "{topic}".
Classify each document as *related* (True) or *not related* (False) with the topic "{topic}".
Produce a list of exactly {n} bools, one for each document, in the same order as the documents.

{"\n".join(f"{i}. {t}" for i, t in enumerate(texts))}
"""

    messages = [{'role': 'user', 'content': prompt}]
    list_model = RootModel[Annotated[List[bool], Field(min_length=n, max_length=n)]]
    model_instances = send_llm_request(messages, response_format=list_model)

    return model_instances.root


if __name__ == '__main__':
    topic = 'machine learning'
    n = 3

    class AcademicPublication(BaseModel):
        """An academic publication contains a title and short abstract (~1 sentence)."""
        title: str
        abstract: str

    publications = generate_documents(topic, n, AcademicPublication)

    print(publications)

    evaluations = evaluate_documents(topic, publications)

    print(evaluations)
