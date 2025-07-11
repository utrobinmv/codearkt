import pytest

from codearkt.codeact import CodeActAgent, Prompts
from codearkt.llm import ChatMessage, LLM


@pytest.mark.asyncio(loop_scope="session")
async def test_codeact_no_tools(gpt_4o_mini: LLM) -> None:
    prompts = Prompts.load("codearkt/prompts/codeact.yaml")
    agent = CodeActAgent(
        name="agent",
        description="Just agent",
        llm=gpt_4o_mini,
        prompts=prompts,
        server_url=None,
    )
    result = await agent.ainvoke(
        [ChatMessage(role="user", content="What is 432412421249 * 4332144219?")],
        session_id="test",
    )
    str_result = str(result).replace(",", "").replace(".", "").replace(" ", "")
    assert "1873272970937648109531" in str_result, result
