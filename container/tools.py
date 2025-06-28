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

_tool_schemas: Dict[str, Tool] = {}


async def _acall(tool: str, *args: Any, **kwargs: Any) -> ToolReturnType:
    async with streamablehttp_client(MCP_URL) as (
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

            _tool_schemas = {tool.name: tool for tool in tools}

            for tool in tools:
                tool_fn: Callable[..., ToolReturnType] = functools.partial(_call, tool.name)
                final_tools[tool.name] = tool_fn
            return final_tools
