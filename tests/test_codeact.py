import asyncio
from textwrap import dedent

from codearkt.codeact import CodeActAgent, extract_code_from_text
from codearkt.llm import ChatMessage, LLM
from codearkt.event_bus import AgentEventBus, EventType
from codearkt.util import get_unique_id

from tests.conftest import MCPServerTest, get_nested_agent


class TestExtractCodeFromText:
    def test_extract_code_from_text_basic(self) -> None:
        text = 'Code:\n```python\nprint("Hello, world!")\n```'
        code = extract_code_from_text(dedent(text))
        assert code == 'print("Hello, world!")', code

    def test_extract_code_from_text_line_breaks(self) -> None:
        text = 'Code:\n```python\n\n\nprint("Hello, world!")\n```'
        code = extract_code_from_text(dedent(text))
        assert code == 'print("Hello, world!")', code

    def test_extract_code_from_text_py(self) -> None:
        text = 'Code:\n```py\nprint("Hello, world!")\n```'
        code = extract_code_from_text(dedent(text))
        assert code == 'print("Hello, world!")', code

    def test_extract_code_from_text_change(self) -> None:
        text = 'Execute code:\n```python\nprint("Hello, world!")\n```'
        code = extract_code_from_text(dedent(text))
        assert code == 'print("Hello, world!")', code

    def test_extract_code_from_text_code_example(self) -> None:
        text = 'Code example:\n```python\nprint("Hello, world!")\n```'
        code = extract_code_from_text(dedent(text))
        assert code is None

    def test_extract_code_from_text_multiple(self) -> None:
        text = "Code:\n```py\na = 1\n```\nAnother code:\n```py\nb = 2\n```"
        code = extract_code_from_text(dedent(text))
        assert code == "a = 1\n\nb = 2", code

    def test_extract_code_from_text_code_unclosed(self) -> None:
        text = 'Code:\n```python\nprint("Hello, world!")\n'
        code = extract_code_from_text(dedent(text))
        assert code is None


class TestCodeActAgent:
    async def test_codeact_no_tools(self, deepseek: LLM) -> None:
        agent = CodeActAgent(
            name="agent",
            description="Just agent",
            llm=deepseek,
            server_url=None,
            tool_names=[],
        )
        result = await agent.ainvoke(
            [ChatMessage(role="user", content="What is 432412421249 * 4332144219?")],
            session_id=get_unique_id(),
        )
        str_result = str(result).replace(",", "").replace(".", "").replace(" ", "")
        assert "1873272970937648109531" in str_result, result

    async def test_codeact_max_iterations(
        self, deepseek: LLM, mcp_server_test: MCPServerTest
    ) -> None:
        _ = mcp_server_test
        agent = CodeActAgent(
            name="agent",
            description="Just agent",
            llm=deepseek,
            tool_names=["arxiv_search"],
            max_iterations=1,
        )
        result = await agent.ainvoke(
            [ChatMessage(role="user", content="Get the exact title of 2409.06820")],
            session_id=get_unique_id(),
        )
        assert "role-playing language models" in str(result).lower(), result

    async def test_codeact_initial_plan(
        self, deepseek: LLM, mcp_server_test: MCPServerTest
    ) -> None:
        _ = mcp_server_test
        agent = CodeActAgent(
            name="agent",
            description="Just agent",
            llm=deepseek,
            tool_names=["arxiv_search"],
            planning_interval=5,
        )
        result = await agent.ainvoke(
            [ChatMessage(role="user", content="Get the exact title of 2409.06820")],
            session_id=get_unique_id(),
        )
        assert "role-playing language models" in str(result).lower(), result

    async def test_codeact_zero_iterations(
        self, deepseek: LLM, mcp_server_test: MCPServerTest
    ) -> None:
        _ = mcp_server_test
        agent = CodeActAgent(
            name="agent",
            description="Just agent",
            llm=deepseek,
            tool_names=["arxiv_search"],
            max_iterations=0,
        )
        result = await agent.ainvoke(
            [ChatMessage(role="user", content="Get the exact title of 2409.06820")],
            session_id=get_unique_id(),
        )
        assert "role-playing language models" not in str(result).lower(), result

    async def test_codeact_images(self, gpt_4o: LLM, mcp_server_test: MCPServerTest) -> None:
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
            session_id=get_unique_id(),
        )
        assert "Player" in str(result), result

    async def test_codeact_multi_agent(self, deepseek: LLM, mcp_server_test: MCPServerTest) -> None:
        _ = mcp_server_test
        agent = CodeActAgent(
            name="agent",
            description="Just agent",
            llm=deepseek,
            managed_agents=[get_nested_agent()],
        )
        query = "Get the exact title of 2409.06820v4."
        result = await agent.ainvoke(
            [
                ChatMessage(
                    role="user",
                    content=query,
                )
            ],
            session_id=get_unique_id(),
        )
        assert "role-playing language models" in str(result).lower(), result

    async def test_codeact_event_bus_simple(self, deepseek: LLM) -> None:
        agent_name = "agent"
        agent = CodeActAgent(
            name=agent_name,
            description="Just agent",
            llm=deepseek,
        )
        query = "What is 432412421249 * 4332144219?"
        event_bus = AgentEventBus()
        session_id = get_unique_id()
        task = asyncio.create_task(
            agent.ainvoke(
                [
                    ChatMessage(
                        role="user",
                        content=query,
                    )
                ],
                session_id=session_id,
                event_bus=event_bus,
            )
        )
        event_bus.register_task(
            session_id=session_id,
            agent_name=agent.name,
            task=task,
        )
        events = []
        async for event in event_bus.stream_events(session_id):
            events.append(event)

        assert len(events) > 0, events

        assert events[0].event_type == EventType.AGENT_START, events[0]
        assert events[0].agent_name == agent_name, events[0]
        assert events[0].session_id == session_id, events[0]
        assert events[0].content is None, events[0]

        assert events[-1].event_type == EventType.AGENT_END, events[-1]
        assert events[-1].agent_name == agent_name, events[-1]
        assert events[-1].session_id == session_id, events[-1]
        assert events[-1].content is None, events[-1]

        assert events[1].event_type == EventType.OUTPUT, events[1]
        assert events[1].agent_name == agent_name, events[1]
        assert events[1].session_id == session_id, events[1]
        assert events[1].content is not None, events[1]

        event_types = {event.event_type for event in events}
        assert EventType.TOOL_CALL in event_types, event_types
        assert EventType.TOOL_RESPONSE in event_types, event_types

        contents = [e.content for e in events if e.event_type == EventType.OUTPUT if e.content]
        final_text = "".join(contents)
        str_result = str(final_text).replace(",", "").replace(".", "").replace(" ", "")
        assert "1873272970937648109531" in str_result, str_result
