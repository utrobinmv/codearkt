import requests
import fire  # type: ignore
from typing import Iterator


HEADERS = {"Content-Type": "application/json", "Accept": "text/event-stream"}


def query_manager_agent(
    query: str,
    base_url: str = "http://localhost:5055",
) -> Iterator[str]:
    url = f"{base_url}/agents/manager"
    payload = {"query": query, "stream": True}

    response = requests.post(url, json=payload, headers=HEADERS, stream=True, timeout=30)
    response.raise_for_status()
    for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
        if chunk:
            yield chunk


DEFAULT_QUERY = "Call librarian agent to find an abstract of the PingPong paper by Ilya Gusev"


def main(query: str = DEFAULT_QUERY) -> None:
    for chunk in query_manager_agent(query):
        print(chunk, end="", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
