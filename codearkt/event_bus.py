import asyncio
from typing import Any, Dict, Optional
from enum import StrEnum

from pydantic import BaseModel


class EventType(StrEnum):
    AGENT_START = "agent_start"
    TOOL_CALL = "tool_call"
    OUTPUT = "output"
    TOOL_RESPONSE = "observation"
    AGENT_END = "agent_end"


class AgentEvent(BaseModel):  # type: ignore
    session_id: str
    agent_name: str
    timestamp: str
    event_type: str
    content: Optional[str] = None


class AgentEventBus:
    def __init__(self) -> None:
        self.subscribers: Dict[str, asyncio.Queue[AgentEvent]] = {}
        self.running_tasks: Dict[str, asyncio.Task[Any]] = {}

    def register_task(self, session_id: str, task: asyncio.Task[Any] | None) -> None:
        if task is not None:
            self.running_tasks[session_id] = task

    async def publish_event(self, event: AgentEvent) -> None:
        if event.session_id not in self.subscribers:
            self.subscribers[event.session_id] = asyncio.Queue()
        await self.subscribers[event.session_id].put(event)

    def subscribe_to_session(self, session_id: str) -> asyncio.Queue[AgentEvent]:
        if session_id not in self.subscribers:
            self.subscribers[session_id] = asyncio.Queue()
        return self.subscribers[session_id]
