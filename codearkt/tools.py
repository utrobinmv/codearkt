from typing import List
import httpx

from mcp import ClientSession, Tool
from mcp.client.streamable_http import streamablehttp_client


async def fetch_tools(url: str) -> List[Tool]:
    all_tools = []
    async with streamablehttp_client(url + "/mcp") as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools_response = await session.list_tools()
            all_tools.extend(tools_response.tools)

    async with httpx.AsyncClient() as client:
        resp = await client.get(url + "/agents/list")
        resp.raise_for_status()
        agent_cards = resp.json()
        print("AGENT CARDS")
        print(agent_cards)
        for card in agent_cards:
            all_tools.append(
                Tool(
                    name="agent__" + card["name"],
                    description=card["description"],
                    inputSchema={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                )
            )
    print("ALL TOOLS")
    print(all_tools)
    return all_tools
