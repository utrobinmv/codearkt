import os
import asyncio
import functools
import httpx
import traceback
from typing import List, Dict, Callable, Any

from mcp import ClientSession, Tool
from mcp.types import ContentBlock
from mcp.client.streamable_http import streamablehttp_client

SERVER_PORT = os.getenv("SERVER_PORT", "5055")
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", 600))
SERVER_URL = f"http://host.docker.internal:{SERVER_PORT}"


ToolReturnType = List[ContentBlock] | str | None

_tool_schemas: Dict[str, Tool] = {}


async def _acall(tool: str, *args: Any, **kwargs: Any) -> ToolReturnType:
    async with streamablehttp_client(SERVER_URL + "/mcp") as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            arguments = dict(kwargs)
            if args and tool in _tool_schemas:
                tool_schema = _tool_schemas[tool]
                if hasattr(tool_schema, "inputSchema") and tool_schema.inputSchema:
                    if (
                        isinstance(tool_schema.inputSchema, dict)
                        and "properties" in tool_schema.inputSchema
                    ):
                        param_names = list(tool_schema.inputSchema["properties"].keys())
                    else:
                        param_names = []
                    for i, arg in enumerate(args):
                        if i < len(param_names):
                            param_name = param_names[i]
                            if param_name not in arguments:
                                arguments[param_name] = arg

            result = await session.call_tool(tool, arguments)
            content_blocks: List[ContentBlock] = result.content
            if len(content_blocks) == 0:
                return None
            if len(content_blocks) == 1 and content_blocks[0].type == "text":
                return content_blocks[0].text
            return content_blocks


def _call(tool: str, *args: Any, **kwargs: Any) -> List[ContentBlock] | str | None:
    return asyncio.run(_acall(tool, *args, **kwargs))


async def fetch_tools() -> Dict[str, Callable[..., ToolReturnType]]:
    global _tool_schemas
    final_tools = {}
    async with streamablehttp_client(SERVER_URL + "/mcp") as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools_response = await session.list_tools()
            tools: List[Tool] = tools_response.tools

            _tool_schemas = {tool.name: tool for tool in tools}

            for tool in tools:
                tool_fn: Callable[..., ToolReturnType] = functools.partial(_call, tool.name)
                final_tools[tool.name] = tool_fn

    agent_cards = []
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(SERVER_URL + "/agents/list")
            response.raise_for_status()
            agent_cards = response.json()
    except Exception:
        print("Failed to fetch agents")
        print(traceback.format_exc())
        pass

    for card in agent_cards:
        agent_name = card["name"]
        url = SERVER_URL + f"/agents/{agent_name}"

        def create_call_agent(url: str) -> Callable[..., Any]:
            def _call_agent(query: str, session_id: str) -> Any:
                payload = {
                    "messages": [{"role": "user", "content": query}],
                    "session_id": session_id,
                    "stream": False,
                }
                resp = httpx.post(url, json=payload, timeout=AGENT_TIMEOUT)
                resp.raise_for_status()
                return resp.json()

            return _call_agent

        final_tools["agent__" + agent_name] = create_call_agent(url)

    return final_tools
