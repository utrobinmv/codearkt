import asyncio
from typing import Any, Dict, Optional
from enum import StrEnum
from dataclasses import dataclass


class EventType(StrEnum):
    STEP_START = "step_start"
    STEP_COMPLETE = "step_complete"
    TOOL_CALL = "tool_call"
    ERROR = "error"
    OUTPUT = "output"
    OBSERVATION = "observation"
    SESSION_END = "session_end"


@dataclass
class AgentEvent:
    session_id: str
    agent_name: str
    timestamp: str
    event_type: str
    data: Dict[str, Any]
    step_number: Optional[int] = None


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
