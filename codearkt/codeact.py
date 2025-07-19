import re
import copy
import uuid
from pathlib import Path
from dataclasses import dataclass
from typing import List, Self, Dict, Any, Optional, Sequence

import yaml

from mcp import Tool
from jinja2 import Template

from codearkt.python_executor import PythonExecutor
from codearkt.tools import fetch_tools
from codearkt.event_bus import AgentEventBus, EventType
from codearkt.llm import LLM, ChatMessages, ChatMessage, FunctionCall, ToolCall


END_CODE_SEQUENCE = "<end_code>"
END_PLAN_SEQUENCE = "<end_plan>"
STOP_SEQUENCES = [END_CODE_SEQUENCE, "Observation:", "Calling tools:"]
DEFAULT_SERVER_URL = "http://localhost:5055"
CURRENT_DIR = Path(__file__).parent


def extract_code_from_text(text: str) -> str | None:
    pattern = r"[C|c]ode[\*]*:\n*```(?:py|python)?\s*\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return "\n\n".join(match.strip() for match in matches)
    return None


def fix_code_actions(messages: List[ChatMessage]) -> List[ChatMessage]:
    for message in messages:
        if message.role != "assistant":
            continue
        if not message.tool_calls:
            continue
        for tool_call in message.tool_calls:
            if tool_call.function.name != "python_interpreter":
                continue
            code_action = str(tool_call.function.arguments).strip()
            if not code_action.startswith("```"):
                code_action = code_action.lstrip("`")
                code_action = "```" + code_action
            if not code_action.endswith("```"):
                code_action = code_action.rstrip("`")
                code_action += "```"
            code_action = code_action.strip() + END_CODE_SEQUENCE
            if isinstance(message.content, str):
                message.content += "\n" + code_action
            else:
                message.content.append({"text": code_action})
        message.tool_calls = []
    return messages


@dataclass
class Prompts:
    system: Template
    final: Template
    initial_plan: Optional[Template] = None
    plan_prefix: Optional[Template] = None

    @classmethod
    def load(cls, path: str | Path) -> Self:
        with open(path) as f:
            template = f.read()
        templates: Dict[str, Any] = yaml.safe_load(template)
        wrapped_templates = {}
        for key, value in templates.items():
            wrapped_templates[key] = Template(value)
        return cls(**wrapped_templates)

    @classmethod
    def default(cls) -> Self:
        return cls.load(CURRENT_DIR / "prompts" / "default.yaml")


class CodeActAgent:
    def __init__(
        self,
        name: str,
        description: str,
        llm: LLM,
        tool_names: Sequence[str] = tuple(),
        prompts: Optional[Prompts] = None,
        max_iterations: int = 10,
        planning_interval: Optional[int] = None,
        server_url: Optional[str] = DEFAULT_SERVER_URL,
        managed_agents: Optional[List[Self]] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.llm: LLM = llm
        self.prompts: Prompts = prompts or Prompts.default()
        self.tool_names = list(tool_names)
        self.max_iterations = max_iterations
        self.planning_interval = planning_interval
        self.server_url = server_url
        self.managed_agents: Optional[List[Self]] = managed_agents
        if self.managed_agents:
            for agent in self.managed_agents:
                agent_tool_name = "agent__" + agent.name
                if agent_tool_name not in self.tool_names:
                    self.tool_names.append(agent_tool_name)

    def get_all_agents(self) -> List[Self]:
        agents = [self]
        if self.managed_agents:
            agents.extend(self.managed_agents)
            for agent in self.managed_agents:
                agents.extend(agent.get_all_agents())
        named_agents = {agent.name: agent for agent in agents}
        return list(named_agents.values())

    async def ainvoke(
        self,
        messages: ChatMessages,
        session_id: str,
        event_bus: AgentEventBus | None = None,
    ) -> str:
        messages = copy.deepcopy(messages)
        await self._publish_event(event_bus, session_id, EventType.AGENT_START)
        python_executor = PythonExecutor(session_id=session_id, tool_names=self.tool_names)

        tools = []
        if self.server_url:
            tools = await fetch_tools(self.server_url)
            tools = [tool for tool in tools if tool.name in self.tool_names]
        system_prompt = self.prompts.system.render(tools=tools)

        messages = fix_code_actions(messages)
        messages = [ChatMessage(role="system", content=system_prompt)] + messages

        for step_number in range(1, self.max_iterations + 1):
            if self.planning_interval is not None and (
                step_number == 1 or (step_number - 1) % self.planning_interval == 0
            ):
                new_messages = await self._run_planning_step(messages, tools, session_id, event_bus)
                messages.extend(new_messages)

            new_messages = await self._step(messages, python_executor, session_id, event_bus)
            messages.extend(new_messages)
            if messages[-1].role == "assistant":
                break
        else:
            new_messages = await self._handle_final_message(messages, session_id, event_bus)
            messages.extend(new_messages)

        python_executor.cleanup()
        await self._publish_event(event_bus, session_id, EventType.AGENT_END)
        return str(messages[-1].content)

    async def _step(
        self,
        messages: ChatMessages,
        python_executor: PythonExecutor,
        session_id: str,
        event_bus: AgentEventBus | None = None,
    ) -> ChatMessages:
        output_text = ""
        output_stream = self.llm.astream(messages, stop=STOP_SEQUENCES)
        tool_call_id = f"toolu_{str(uuid.uuid4())[:8]}"
        async for event in output_stream:
            if isinstance(event.content, str):
                chunk = event.content
            elif isinstance(event.content, list):
                chunk = "\n".join([str(item) for item in event.content])
            output_text += chunk
            await self._publish_event(event_bus, session_id, EventType.OUTPUT, chunk)
        await self._publish_event(event_bus, session_id, EventType.OUTPUT, "\n")

        if (
            output_text
            and output_text.strip().endswith("```")
            and not output_text.strip().endswith(END_CODE_SEQUENCE)
        ):
            chunk = END_CODE_SEQUENCE + "\n"
            output_text += chunk

        code_action = extract_code_from_text(output_text)
        new_messages = []
        if code_action is None:
            new_messages.append(ChatMessage(role="assistant", content=output_text))
            return new_messages

        tool_call_message = ChatMessage(
            role="assistant",
            content=output_text,
            tool_calls=[
                ToolCall(
                    id=tool_call_id,
                    function=FunctionCall(name="python_interpreter", arguments=code_action),
                )
            ],
        )
        await self._publish_event(event_bus, session_id, EventType.TOOL_CALL, code_action)
        new_messages.append(tool_call_message)
        try:
            code_result = await python_executor.invoke(code_action)
            code_result_messages = code_result.to_messages(tool_call_id)
            new_messages.extend(code_result_messages)
            tool_output: str = str(code_result_messages[0].content) + "\n"
            await self._publish_event(event_bus, session_id, EventType.TOOL_RESPONSE, tool_output)
        except Exception as e:
            new_messages.append(
                ChatMessage(role="tool", content=f"Error: {e}", tool_call_id=tool_call_id)
            )
            await self._publish_event(
                event_bus, session_id, EventType.TOOL_RESPONSE, f"Error: {e}\n"
            )
        return new_messages

    async def _handle_final_message(
        self,
        messages: ChatMessages,
        session_id: str,
        event_bus: AgentEventBus | None = None,
    ) -> ChatMessages:
        prompt: str = self.prompts.final.render()
        final_message = ChatMessage(role="user", content=prompt)
        output_stream = self.llm.astream(messages + [final_message], stop=STOP_SEQUENCES)
        output_text = ""
        async for event in output_stream:
            if isinstance(event.content, str):
                chunk = event.content
            elif isinstance(event.content, list):
                chunk = "\n".join([str(item) for item in event.content])
            output_text += chunk
            await self._publish_event(event_bus, session_id, EventType.OUTPUT, chunk)
        return [ChatMessage(role="assistant", content=output_text)]

    async def _run_planning_step(
        self,
        messages: ChatMessages,
        tools: List[Tool],
        session_id: str,
        event_bus: AgentEventBus | None = None,
    ) -> ChatMessages:
        assert (
            self.prompts.initial_plan is not None
        ), "Planning prompt is not set, but planning is enabled"
        assert (
            self.prompts.plan_prefix is not None
        ), "Plan prefix is not set, but planning is enabled"

        conversation = "\n\n".join([f"{m.role}: {m.content}" for m in messages])
        planning_prompt = self.prompts.initial_plan.render(conversation=conversation, tools=tools)
        input_messages = [ChatMessage(role="user", content=planning_prompt)]

        output_stream = self.llm.astream(input_messages, stop=[END_PLAN_SEQUENCE])

        plan_prefix = self.prompts.plan_prefix.render().strip() + "\n\n"
        await self._publish_event(event_bus, session_id, EventType.OUTPUT, plan_prefix)
        output_text = plan_prefix

        async for event in output_stream:
            assert isinstance(event.content, str)
            chunk = event.content
            output_text += chunk
            await self._publish_event(event_bus, session_id, EventType.OUTPUT, chunk)
        await self._publish_event(event_bus, session_id, EventType.OUTPUT, "\n\n")

        return [ChatMessage(role="assistant", content=output_text)]

    async def _publish_event(
        self,
        event_bus: AgentEventBus | None,
        session_id: str,
        event_type: EventType = EventType.OUTPUT,
        content: Optional[str] = None,
    ) -> None:
        if not event_bus:
            return
        await event_bus.publish_event(
            session_id=session_id,
            agent_name=self.name,
            event_type=event_type,
            content=content,
        )
