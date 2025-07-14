import requests
import fire  # type: ignore
from typing import Iterator


from codearkt.event_bus import AgentEvent


HEADERS = {"Content-Type": "application/json", "Accept": "text/event-stream"}


def query_manager_agent(
    query: str,
    base_url: str = "http://localhost:5055",
) -> Iterator[AgentEvent]:
    url = f"{base_url}/agents/manager"
    payload = {"query": query, "stream": True}

    response = requests.post(url, json=payload, headers=HEADERS, stream=True, timeout=30)
    response.raise_for_status()
    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
        if chunk:
            event = AgentEvent.model_validate_json(chunk)
            yield event


DEFAULT_QUERY = "Find an abstract of the PingPong paper by Ilya Gusev"


def main(query: str = DEFAULT_QUERY) -> None:
    for event in query_manager_agent(query):
        if event.content:
            print(event.content, end="", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
