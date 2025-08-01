import asyncio
import socket
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


def find_free_port() -> Optional[int]:
    for port in range(5000, 6001):
        try:
            with socket.socket() as s:
                s.bind(("", port))
                return port
        except Exception:
            continue
    return None


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


def create_agent_endpoint(
    agent_app: FastAPI,
    agent_instance: CodeActAgent,
    server_host: str,
    server_port: int,
) -> Callable[..., Any]:
    @agent_app.post(f"/{agent_instance.name}")  # type: ignore
    async def agent_tool(request: AgentRequest) -> Any:
        session_id = request.session_id or get_unique_id()

        if request.stream:

            async def stream_response() -> AsyncGenerator[str, None]:
                task = asyncio.create_task(
                    agent_instance.ainvoke(
                        messages=request.messages,
                        session_id=session_id,
                        event_bus=event_bus,
                        server_host=server_host,
                        server_port=server_port,
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


def get_agent_app(main_agent: CodeActAgent, server_host: str, server_port: int) -> FastAPI:
    agent_app = FastAPI(
        title="CodeArkt Agent App", description="Agent app for CodeArkt", version="1.0.0"
    )

    agent_cards = []
    for agent in main_agent.get_all_agents():
        agent_cards.append(AgentCard(name=agent.name, description=agent.description))
        create_agent_endpoint(agent_app, agent, server_host, server_port)

    async def cancel_session(request: CancelRequest) -> Dict[str, str]:
        event_bus.cancel_session(request.session_id)
        return {"status": "cancelled", "session_id": request.session_id}

    async def get_agents() -> List[AgentCard]:
        return agent_cards

    agent_app.get("/list")(get_agents)
    agent_app.post("/cancel")(cancel_session)
    return agent_app


def get_mcp_app(mcp_config: Optional[Dict[str, Any]]) -> Optional[FastAPI]:
    if not mcp_config:
        return None
    proxy = FastMCP.as_proxy(mcp_config, name="Codearkt MCP Proxy")
    return proxy.http_app()


def get_main_app(
    agent: CodeActAgent,
    mcp_config: Optional[Dict[str, Any]] = None,
    server_host: str = "0.0.0.0",
    server_port: int = 5055,
) -> FastAPI:
    agent_app = get_agent_app(agent, server_host, server_port)
    mcp_app = get_mcp_app(mcp_config) or FastAPI()
    mcp_app.mount("/agents", agent_app)
    return mcp_app


def run_server(
    agent: CodeActAgent, mcp_config: Dict[str, Any], host: str = "0.0.0.0", port: int = 5055
) -> None:
    app = get_main_app(
        agent=agent,
        mcp_config=mcp_config,
        server_host=host,
        server_port=port,
    )
    uvicorn.run(
        app,
        host=host,
        port=port,
        access_log=False,
        lifespan="on",
        ws="none",
    )


async def run_query(
    query: str, agent: CodeActAgent, mcp_config: Optional[Dict[str, Any]] = None
) -> str:
    host = "0.0.0.0"
    port = find_free_port()
    assert port is not None
    app = get_main_app(agent, mcp_config, server_host=host, server_port=port)

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="error",
        access_log=False,
        lifespan="on",
        ws="none",
    )
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    try:
        await asyncio.wait_for(_wait_until_started(server), timeout=30)
        result = await agent.ainvoke(
            [ChatMessage(role="user", content=query)],
            session_id=get_unique_id(),
            server_host=host,
            server_port=port,
        )
    finally:
        server.should_exit = True
        await server_task

    return result


async def _wait_until_started(server: uvicorn.Server) -> None:
    while not server.started:
        await asyncio.sleep(0.05)
