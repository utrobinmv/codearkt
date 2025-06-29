from typing import List, Dict, Any

from fastmcp import FastMCP

from codearkt.codeact import CodeActAgent, ChatMessage


def run_mcp_server(agents: List[CodeActAgent], mcp_config: Dict[str, Any]) -> None:
    proxy = FastMCP.as_proxy(mcp_config, name="Codearkt MCP Proxy", stateless_http=True)
    for agent in agents:

        @proxy.tool(name=agent.name, description=agent.description)
        async def agent_tool(query: str) -> str:
            await agent.ainvoke([ChatMessage(role="user", content=query)])
            return agent.messages[-1].content

        proxy.add_tool(agent_tool)

    proxy.run(transport="http", host="0.0.0.0", port=5055, path="/mcp")
