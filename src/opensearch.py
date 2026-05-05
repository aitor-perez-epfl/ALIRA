from typing import Optional, Union

from opensearchpy import OpenSearch

from config import config

EMBEDDING_MODEL_ID = "41_9BZ0BrpQLIV_1Av_A"

client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_auth=(config["OPENSEARCH_USER"], config["OPENSEARCH_PASSWORD"]),
    use_ssl=True,
    verify_certs=False,
    ssl_show_warn=False,
)

def embed(texts: list[str]) -> list[list[float]]:
    response = client.plugins.ml.predict(
        algorithm_name="text_embedding",
        model_id=EMBEDDING_MODEL_ID,
        body={
            "text_docs": texts,
            "target_response": ["sentence_embedding"]
        }
    )

    results = [r["output"][0]["data"] for r in response["inference_results"]]

    return results

def fetch_all(index_name: str, document_type: Optional[Union[str, list[str]]] = None):
    if not document_type:
        document_type = []
    elif isinstance(document_type, str):
        document_type = [document_type]

    size = client.count(index=index_name)["count"]

    # We need to paginate if size > page_size
    page_size = 10_000
    needs_pagination = size > page_size

    if document_type:
        query = {"terms": {"type": document_type}}
    else:
        query = {"match_all": {}}

    body = {
        "_source": {"includes": ["id", "type", "name", "text", "embedding"]},
        "size": page_size if needs_pagination else size,
        "query": query,
        "sort": [{"_score": "desc"}, {"_id": "asc"}],
    }

    # Point-in-time pagination
    pit_id = client.create_pit(index=index_name, keep_alive="5m")["pit_id"]
    body["pit"] = {"id": pit_id, "keep_alive": "5m"}
    all_hits = []

    try:
        remaining = size
        while True:
            # Fetch next batch
            body["size"] = min(page_size, remaining)
            response = client.search(body=body)  # no index= when using PIT

            # Break if no hits
            hits = response["hits"]["hits"]
            if not hits:
                break

            # Store new hits
            all_hits.extend(hits)

            # Refresh pit_id
            pit_id = response.get("pit_id", pit_id)
            body["pit"]["id"] = pit_id
            body["search_after"] = hits[-1]["sort"]

            # Update remaining documents
            remaining -= len(hits)

            # Break if no more documents
            if remaining <= 0 or len(hits) < body["size"]:
                break
    finally:
        # Clear pit
        client.delete_pit(body={"pit_id": pit_id})

    # Update response and return
    response["hits"]["hits"] = all_hits
    response["hits"]["total"] = {"value": len(all_hits), "relation": "eq"}

    return response

def search(index_name: str, text: Union[str, list[str]], document_type: Optional[Union[str, list[str]]] = None, size: int = 10):
    if isinstance(text, str):
        text = [text]

    if not document_type:
        document_type = []
    elif isinstance(document_type, str):
        document_type = [document_type]

    body = {
        "_source": {"includes": ["id", "type", "name", "text", "embedding"]},
        "size": size,
        "query": {
            "bool": {
                "should": [{
                    "dis_max": {
                        "queries": [
                            {
                                "neural": {
                                    "embedding": {
                                        "query_text": query_text,
                                        "model_id": EMBEDDING_MODEL_ID,
                                        "k": size,
                                    }
                                }
                            }
                            for query_text in text
                        ]
                    }
                }]
            }
        },
    }

    if document_type:
        body["query"]["bool"]["filter"] = [{"terms": {"type": document_type}}]

    return client.search(index=index_name, body=body)


if __name__ == "__main__":
    # r = fetch_all(index_name='test2', document_type='publication')
    # print(len(r["hits"]["hits"]))

    r = search(index_name='test2', text='robotics', document_type='publication')
    for hit in r["hits"]["hits"]:
        print(hit["_source"])
