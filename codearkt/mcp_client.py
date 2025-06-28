from typing import List

from mcp import ClientSession, Tool
from mcp.client.streamable_http import streamablehttp_client


async def fetch_tools(url: str) -> List[Tool]:
    async with streamablehttp_client(url) as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools_response = await session.list_tools()
            return tools_response.tools
