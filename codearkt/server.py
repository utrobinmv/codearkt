import asyncio
from typing import Dict, Any, Optional, List, Callable, AsyncGenerator

import uvicorn
from fastmcp import FastMCP, settings as fastmcp_settings
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from codearkt.codeact import CodeActAgent
from codearkt.llm import ChatMessage
from codearkt.event_bus import AgentEventBus
from codearkt.util import get_unique_id


event_bus = AgentEventBus()
fastmcp_settings.stateless_http = True


class AgentRequest(BaseModel):  # type: ignore
    messages: List[ChatMessage]
    session_id: Optional[str] = None
    stream: bool = False


class AgentCard(BaseModel):  # type: ignore
    name: str
    description: str


AGENT_RESPONSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def create_agent_endpoint(agent_app: FastAPI, agent_instance: CodeActAgent) -> Callable[..., Any]:
    @agent_app.post(f"/{agent_instance.name}")  # type: ignore
    async def agent_tool(request: AgentRequest) -> Any:
        session_id = request.session_id or get_unique_id()

        if request.stream:

            async def stream_response() -> AsyncGenerator[str, None]:
                task = asyncio.create_task(
                    agent_instance.ainvoke(
                        messages=request.messages, session_id=session_id, event_bus=event_bus
                    )
                )
                event_bus.register_task(
                    session_id=session_id,
                    agent_name=agent_instance.name,
                    task=task,
                )
                try:
                    async for event in event_bus.stream_events(session_id):
                        yield event.model_dump_json()
                finally:
                    event_bus.cancel_session(session_id)

            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
                headers=AGENT_RESPONSE_HEADERS,
            )
        else:
            result = await agent_instance.ainvoke(
                messages=request.messages, session_id=session_id, event_bus=event_bus
            )
            return result

    return agent_tool  # type: ignore


class CancelRequest(BaseModel):  # type: ignore
    session_id: str


def get_agent_app(main_agent: CodeActAgent) -> FastAPI:
    agent_app = FastAPI(
        title="CodeArkt Agent App", description="Agent app for CodeArkt", version="1.0.0"
    )

    agent_cards = []
    for agent in main_agent.get_all_agents():
        agent_cards.append(AgentCard(name=agent.name, description=agent.description))
        create_agent_endpoint(agent_app, agent)

    async def cancel_session(request: CancelRequest) -> Dict[str, str]:
        event_bus.cancel_session(request.session_id)
        return {"status": "cancelled", "session_id": request.session_id}

    async def get_agents() -> List[AgentCard]:
        return agent_cards

    agent_app.get("/list")(get_agents)
    agent_app.post("/cancel")(cancel_session)
    return agent_app


def get_mcp_app(mcp_config: Dict[str, Any]) -> FastAPI:
    proxy = FastMCP.as_proxy(mcp_config, name="Codearkt MCP Proxy")
    return proxy.http_app()


def get_main_app(agent: CodeActAgent, mcp_config: Dict[str, Any]) -> FastAPI:
    agent_app = get_agent_app(agent)
    mcp_app = get_mcp_app(mcp_config)
    mcp_app.mount("/agents", agent_app)
    return mcp_app


def run_server(
    agent: CodeActAgent, mcp_config: Dict[str, Any], host: str = "0.0.0.0", port: int = 5055
) -> None:
    app = get_main_app(agent, mcp_config)
    uvicorn.run(
        app,
        host=host,
        port=port,
        access_log=False,
        lifespan="on",
        ws="none",
    )
