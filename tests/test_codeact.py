import pytest

from codearkt.codeact import CodeActAgent
from codearkt.llm import ChatMessage, LLM

from tests.conftest import MCPServerTest, get_nested_agent


@pytest.mark.asyncio(loop_scope="function")
async def test_codeact_no_tools(deepseek: LLM) -> None:
    agent = CodeActAgent(
        name="agent",
        description="Just agent",
        llm=deepseek,
        server_url=None,
        tool_names=[],
    )
    result = await agent.ainvoke(
        [ChatMessage(role="user", content="What is 432412421249 * 4332144219?")],
        session_id="test",
    )
    str_result = str(result).replace(",", "").replace(".", "").replace(" ", "")
    assert "1873272970937648109531" in str_result, result


@pytest.mark.asyncio(loop_scope="function")
async def test_codeact_images(gpt_4o: LLM, mcp_server_test: MCPServerTest) -> None:
    _ = mcp_server_test
    agent = CodeActAgent(
        name="agent",
        description="Just agent",
        llm=gpt_4o,
        tool_names=["show_image"],
    )
    image_url = "https://arxiv.org/html/2409.06820v4/extracted/6347978/pingpong_v3.drawio.png"
    result = await agent.ainvoke(
        [
            ChatMessage(
                role="user",
                content=f"What blocks are in this image? {image_url}\nUse show_image tool",
            )
        ],
        session_id="test",
    )
    assert "Player" in str(result), result


@pytest.mark.asyncio(loop_scope="function")
async def test_multi_agent(deepseek: LLM, mcp_server_test: MCPServerTest) -> None:
    _ = mcp_server_test
    agent = CodeActAgent(
        name="agent",
        description="Just agent",
        llm=deepseek,
        managed_agents=[get_nested_agent()],
    )
    query = "Get the exact abstract of 2409.06820v4."
    result = await agent.ainvoke(
        [
            ChatMessage(
                role="user",
                content=query,
            )
        ],
        session_id="test",
    )
    assert "evaluating the role-playing capabilities" in str(result), result
