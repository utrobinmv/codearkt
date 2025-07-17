from typing import Iterator

import httpx
import fire  # type: ignore

from codearkt.event_bus import AgentEvent


HEADERS = {"Content-Type": "application/json", "Accept": "text/event-stream"}


def query_manager_agent(
    query: str,
    base_url: str = "http://localhost:5055",
) -> Iterator[AgentEvent]:
    url = f"{base_url}/agents/manager"
    payload = {"query": query, "stream": True}

    with httpx.stream("POST", url, json=payload, headers=HEADERS, timeout=60) as response:
        response.raise_for_status()
        for chunk in response.iter_text():
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
