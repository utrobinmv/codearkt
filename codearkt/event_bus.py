import asyncio
from typing import Any, List, Dict, Optional
from enum import StrEnum
from datetime import datetime
from dataclasses import dataclass


class EventType(StrEnum):
    STEP_START = "step_start"
    STEP_COMPLETE = "step_complete"
    TOOL_CALL = "tool_call"
    ERROR = "error"
    OUTPUT = "output"
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
    def __init__(self):
        self.subscribers: Dict[str, asyncio.Queue] = {}
        self.active_sessions: Dict[str, bool] = {}
        self.session_events: Dict[str, List[AgentEvent]] = {}
        self.cancellation_tokens: Dict[str, asyncio.Event] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}

    def register_task(self, session_id: str, task: asyncio.Task):
        self.running_tasks[session_id] = task

    async def publish_event(self, event: AgentEvent):
        if event.session_id not in self.session_events:
            self.session_events[event.session_id] = []
        self.session_events[event.session_id].append(event)

        if event.session_id in self.subscribers:
            try:
                await self.subscribers[event.session_id].put(event)
            except Exception:
                pass

    def subscribe_to_session(self, session_id: str) -> asyncio.Queue:
        if session_id not in self.subscribers:
            self.subscribers[session_id] = asyncio.Queue()
        return self.subscribers[session_id]

    def start_session(self, session_id: str):
        self.active_sessions[session_id] = True

    def stop_session(self, session_id: str):
        self.active_sessions[session_id] = False
        # Send end event
        asyncio.create_task(
            self.publish_event(
                AgentEvent(
                    session_id=session_id,
                    agent_name="system",
                    timestamp=datetime.now().isoformat(),
                    event_type="session_end",
                    data={"reason": "stopped"},
                )
            )
        )

    def cleanup_session(self, session_id: str):
        if session_id in self.subscribers:
            del self.subscribers[session_id]
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
