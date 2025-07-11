import re
import uuid
from pathlib import Path
from dataclasses import dataclass
from typing import List, Self, Dict, Any, Optional
from datetime import datetime

import yaml

from mcp import Tool
from jinja2 import Template

from codearkt.python_executor import PythonExecutor
from codearkt.tools import fetch_tools
from codearkt.event_bus import AgentEventBus, AgentEvent, EventType
from codearkt.llm import LLM, ChatMessages, ChatMessage, FunctionCall, ToolCall


STOP_SEQUENCES = ["<end_code>", "Observation:", "Calling tools:"]
DEFAULT_SERVER_URL = "http://localhost:5055"


def extract_code_from_text(text: str) -> str | None:
    pattern = r"Code:\n*```(?:py|python)?\s*\n(.*?)\n```"
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
            if tool_call.function.name == "python_interpreter":
                code_action = str(tool_call.function.arguments).strip()
                if not code_action.startswith("```"):
                    code_action = code_action.lstrip("`")
                    code_action = "```" + code_action
                if not code_action.endswith("```"):
                    code_action = code_action.rstrip("`")
                    code_action += "```"
                if isinstance(message.content, str):
                    message.content += "\n" + code_action
                else:
                    message.content.append({"text": code_action})
        message.tool_calls = []
    return messages


@dataclass
class Prompts:
    system: str
    final: str

    @classmethod
    def load(cls, path: str | Path) -> Self:
        with open(path) as f:
            template = f.read()
        templates: Dict[str, Any] = yaml.safe_load(template)
        return cls(**templates)

    def format(self, tools: List[Tool]) -> None:
        system_template = Template(self.system)
        self.system = system_template.render(tools=tools)


class CodeActAgent:
    def __init__(
        self,
        name: str,
        description: str,
        llm: LLM,
        prompts: Prompts,
        max_iterations: int = 10,
        server_url: Optional[str] = DEFAULT_SERVER_URL,
        managed_agents: Optional[List[Self]] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.llm: LLM = llm
        self.prompts: Prompts = prompts
        self.max_iterations = max_iterations
        self.server_url = server_url
        self.managed_agents: Optional[List[Self]] = managed_agents
        self.event_bus: Optional[AgentEventBus] = None

    def set_event_bus(self, event_bus: AgentEventBus) -> None:
        self.event_bus = event_bus

    async def ainvoke(
        self,
        messages: ChatMessages,
        session_id: str,
    ) -> str:
        print(f"Invoking agent {self.name} with session_id {session_id}")
        python_executor = PythonExecutor(session_id=session_id)

        tools = []
        if self.server_url:
            tools = await fetch_tools(self.server_url)
        self.prompts.format(tools=tools)

        messages = fix_code_actions(messages)
        messages = [ChatMessage(role="system", content=self.prompts.system)] + messages

        for i in range(self.max_iterations):
            print(self.name, f"step {i} started")
            await self._step(messages, python_executor, session_id)
            if messages[-1].role == "assistant":
                break
        else:
            await self._handle_final_message(messages)
        return str(messages[-1].content)

    async def _step(
        self,
        messages: ChatMessages,
        python_executor: PythonExecutor,
        session_id: str,
    ) -> None:
        output_text = ""
        output_stream = self.llm.astream(messages, stop=STOP_SEQUENCES)
        tool_call_id = f"toolu_{str(uuid.uuid4())[:8]}"
        async for event in output_stream:
            if isinstance(event.content, str):
                chunk = event.content
            elif isinstance(event.content, list):
                chunk = "\n".join([str(item) for item in event.content])
            output_text += chunk
            print(chunk, end="")
            if self.event_bus:
                await self.event_bus.publish_event(
                    AgentEvent(
                        session_id=session_id,
                        agent_name=self.name,
                        timestamp=datetime.now().isoformat(),
                        event_type=EventType.OUTPUT,
                        data={"text": chunk},
                    )
                )

        if (
            output_text
            and output_text.strip().endswith("```")
            and not output_text.strip().endswith("<end_code>")
        ):
            chunk = "<end_code>\n"
            output_text += chunk
            print(chunk)

        code_action = extract_code_from_text(output_text)
        if code_action is None:
            messages.append(ChatMessage(role="assistant", content=output_text))
            return

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
        messages.append(tool_call_message)
        try:
            execution_logs = await python_executor.invoke(code_action)
            observation = "Execution logs:\n" + execution_logs
        except Exception as e:
            observation = f"Error: {e}"
        print("Observation:", observation)
        tool_message = ChatMessage(role="tool", content=observation, tool_call_id=tool_call_id)
        messages.append(tool_message)

    async def _handle_final_message(self, messages: ChatMessages) -> None:
        prompt = self.prompts.final
        final_message = ChatMessage(role="user", content=prompt)
        messages.append(final_message)

        output_stream = self.llm.astream(messages, stop=STOP_SEQUENCES)
        output_text = ""
        async for event in output_stream:
            if isinstance(event.content, str):
                output_text += event.content
        messages.append(ChatMessage(role="assistant", content=output_text))

    def get_all_agents(self) -> List[Self]:
        agents = [self]
        if self.managed_agents:
            agents.extend(self.managed_agents)
            for agent in self.managed_agents:
                agents.extend(agent.get_all_agents())
        named_agents = {agent.name: agent for agent in agents}
        return list(named_agents.values())
