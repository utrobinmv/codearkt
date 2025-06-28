import os
import asyncio
import functools
from typing import List, Dict, Callable, Any

from mcp import ClientSession, Tool
from mcp.types import ContentBlock
from mcp.client.streamable_http import streamablehttp_client

MCP_PORT = os.getenv("MCP_PORT", "5055")
MCP_URL = os.getenv("MCP_URL", f"http://172.17.0.1:{MCP_PORT}/mcp")

ToolReturnType = List[ContentBlock] | str | None


async def _acall(tool: str, **kwargs: Any) -> ToolReturnType:
    async with streamablehttp_client(MCP_URL) as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool, kwargs)
            content_blocks: List[ContentBlock] = result.content
            if len(content_blocks) == 0:
                return None
            if len(content_blocks) == 1 and content_blocks[0].type == "text":
                return content_blocks[0].text
            return content_blocks


def _call(tool: str, **kwargs: Any) -> List[ContentBlock] | str | None:
    return asyncio.run(_acall(tool, **kwargs))


async def fetch_tools() -> Dict[str, Callable[..., ToolReturnType]]:
    async with streamablehttp_client(MCP_URL) as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools_response = await session.list_tools()
            tools: List[Tool] = tools_response.tools
            final_tools = {}
            for tool in tools:
                tool_fn: Callable[..., ToolReturnType] = functools.partial(_call, tool.name)
                final_tools[tool.name] = tool_fn
            return final_tools
