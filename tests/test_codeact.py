import pytest

from codearkt.codeact import CodeActAgent, Prompts, ChatMessage
from codearkt.llm import LLM


@pytest.mark.asyncio(scope="session")
async def test_codeact(gpt_4o_mini: LLM):
    prompts = Prompts.load("codearkt/prompts/codeact.yaml")
    agent = CodeActAgent(
        name="agent",
        description="Just agent",
        llm=gpt_4o_mini,
        prompts=prompts,
    )
    messages = await agent.ainvoke(
        [
            ChatMessage(
                role="user", content="Print the abstract of the PingPong paper by Ilya Gusev"
            )
        ],
        session_id="test",
    )
    assert "a player model" in messages[-1].content, messages
