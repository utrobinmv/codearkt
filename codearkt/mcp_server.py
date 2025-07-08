import asyncio
import uuid
from typing import Dict, Any, Optional

from fastmcp import FastMCP

from codearkt.codeact import CodeActAgent, ChatMessage
from codearkt.event_bus import AgentEventBus


event_bus = AgentEventBus()


def run_mcp_server(main_agent: CodeActAgent, mcp_config: Dict[str, Any]) -> None:
    proxy = FastMCP.as_proxy(mcp_config, name="Codearkt MCP Proxy", stateless_http=True)

    for agent in main_agent.get_all_agents():
        agent.set_event_bus(event_bus)

        async def agent_tool(
            query: str,
            session_id: Optional[str] = None,
            agent_instance=agent
        ) -> str:
            session_id = session_id or str(uuid.uuid4())

            async def run_agent(query: str):
                current_task = asyncio.current_task()
                event_bus.register_task(session_id, current_task)
                return await agent_instance.ainvoke(
                    messages=[ChatMessage(role="user", content=query)],
                    session_id=session_id,
                )

            return await run_agent(query=query)

        proxy.add_tool(
            proxy.tool(name="agent__" + agent.name, description=agent.description)(
                agent_tool
            )
        )

    @proxy.tool(name="get_agent_logs", description="Get the logs for the agent")
    async def agent_logs(session_id: str) -> str:
        queue = event_bus.subscribe_to_session(session_id)
        elements = []
        while not queue.empty():
            event = await asyncio.wait_for(queue.get(), timeout=1.0)
            elements.append(event.data)
        return "".join(elements)

    proxy.run(transport="http", host="0.0.0.0", port=5055, path="/mcp")
