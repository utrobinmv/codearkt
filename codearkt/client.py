from typing import Iterator, List

import httpx

from codearkt.event_bus import AgentEvent
from codearkt.llm import ChatMessage

HEADERS = {"Content-Type": "application/json", "Accept": "text/event-stream"}
BASE_URL = "http://localhost:5055"


def query_agent(
    history: List[ChatMessage],
    *,
    session_id: str | None = None,
    base_url: str = BASE_URL,
    agent_name: str = "manager",
) -> Iterator[AgentEvent]:
    url = f"{base_url}/agents/{agent_name}"
    serialized_history = [m.model_dump() for m in history]
    payload = {"messages": serialized_history, "stream": True}
    if session_id is not None:
        payload["session_id"] = session_id

    timeout = httpx.Timeout(connect=10, pool=None, read=None, write=None)
    with httpx.stream("POST", url, json=payload, headers=HEADERS, timeout=timeout) as response:
        response.raise_for_status()
        for chunk in response.iter_text():
            if chunk:
                yield AgentEvent.model_validate_json(chunk)


def stop_agent(session_id: str, base_url: str = BASE_URL) -> bool:
    try:
        httpx.post(f"{base_url}/agents/cancel", json={"session_id": session_id}, timeout=5.0)
        return True
    except httpx.HTTPError:
        return False
