import re
import copy
import logging
from pathlib import Path
from textwrap import dedent
from dataclasses import dataclass, field
from typing import List, Self, Dict, Any, Optional, Sequence, get_args

import yaml

from mcp import Tool
from jinja2 import Template

from codearkt.python_executor import PythonExecutor
from codearkt.tools import fetch_tools
from codearkt.event_bus import AgentEventBus, EventType
from codearkt.llm import LLM, ChatMessages, ChatMessage
from codearkt.util import get_unique_id

DEFAULT_END_CODE_SEQUENCE = "<end_code>"
DEFAULT_END_PLAN_SEQUENCE = "<end_plan>"
DEFAULT_STOP_SEQUENCES = [DEFAULT_END_CODE_SEQUENCE, "Observation:", "Calling tools:"]
AGENT_TOOL_PREFIX = "agent__"


def extract_code_from_text(text: str) -> str | None:
    pattern = r"[Cc]ode[\*]*\:[\*]*\s*\n*```(?:py|python)?\s*\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return "\n\n".join(match.strip() for match in matches)
    return None


def convert_code_to_content(
    messages: List[ChatMessage], end_code_sequence: str
) -> List[ChatMessage]:
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
            code_action = code_action.strip() + end_code_sequence
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
    plan: Optional[Template] = None
    plan_prefix: Optional[Template] = None
    plan_suffix: Optional[Template] = None
    end_code_sequence: str = DEFAULT_END_CODE_SEQUENCE
    end_plan_sequence: str = DEFAULT_END_PLAN_SEQUENCE
    stop_sequences: List[str] = field(default_factory=lambda: DEFAULT_STOP_SEQUENCES)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        with open(path) as f:
            template = f.read()
        templates: Dict[str, Any] = yaml.safe_load(template)
        wrapped_templates: Dict[str, Any] = {}
        for key, value in templates.items():
            if value is None:
                continue
            field_type = cls.__annotations__.get(key)
            if field_type is Template or Template in get_args(field_type):
                wrapped_templates[key] = Template(value)
            elif field_type is str:
                wrapped_templates[key] = value.strip()
            else:
                wrapped_templates[key] = value
        return cls(**wrapped_templates)

    @classmethod
    def default(cls) -> Self:
        current_dir = Path(__file__).parent
        return cls.load(current_dir / "prompts" / "default.yaml")


class CodeActAgent:
    def __init__(
        self,
        name: str,
        description: str,
        llm: LLM,
        tool_names: Sequence[str] = tuple(),
        prompts: Optional[Prompts] = None,
        max_iterations: int = 10,
        verbosity_level: int = logging.ERROR,
        planning_interval: Optional[int] = None,
        managed_agents: Optional[List[Self]] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.llm: LLM = llm
        self.prompts: Prompts = prompts or Prompts.default()
        self.tool_names = list(tool_names)
        self.max_iterations = max_iterations
        self.verbosity_level = verbosity_level
        self.planning_interval = planning_interval
        self.managed_agents: Optional[List[Self]] = managed_agents

        if self.managed_agents:
            for agent in self.managed_agents:
                agent_tool_name = AGENT_TOOL_PREFIX + agent.name
                if agent_tool_name not in self.tool_names:
                    self.tool_names.append(agent_tool_name)

        self.logger = logging.getLogger(self.__class__.__name__ + ":" + self.name)
        self.logger.setLevel(self.verbosity_level)
        if not self.logger.hasHandlers():
            self.logger.addHandler(logging.StreamHandler())

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
        server_host: Optional[str] = None,
        server_port: Optional[int] = None,
    ) -> str:
        messages = copy.deepcopy(messages)

        run_id = get_unique_id()
        await self._publish_event(event_bus, session_id, EventType.AGENT_START)
        self._log(f"Starting agent {self.name}", run_id=run_id, session_id=session_id)

        python_executor = None
        try:
            python_executor = PythonExecutor(
                session_id=session_id,
                tool_names=self.tool_names,
                interpreter_id=run_id,
                tools_server_port=server_port,
                tools_server_host=server_host,
            )
            self._log("Python interpreter started", run_id=run_id, session_id=session_id)
            self._log(
                f"Host: {server_host}, port: {server_port}",
                run_id=run_id,
                session_id=session_id,
            )

            tools = await self._get_tools(server_host=server_host, server_port=server_port)
            self._log(
                f"Fetched tools: {[tool.name for tool in tools]}",
                run_id=run_id,
                session_id=session_id,
            )
            system_prompt = self.prompts.system.render(tools=tools)

            # Form input messages
            messages = convert_code_to_content(
                messages, end_code_sequence=self.prompts.end_code_sequence
            )
            messages = [ChatMessage(role="system", content=system_prompt)] + messages

            for step_number in range(1, self.max_iterations + 1):
                # Optional planning step
                if self.planning_interval is not None and (
                    step_number == 1 or (step_number - 1) % self.planning_interval == 0
                ):
                    self._log(
                        f"Planning step {step_number} started", run_id=run_id, session_id=session_id
                    )
                    new_messages = await self._run_planning_step(
                        messages, tools, session_id, event_bus
                    )
                    messages.extend(new_messages)
                    self._log(
                        f"Planning step {step_number} completed",
                        run_id=run_id,
                        session_id=session_id,
                    )

                # Main step
                self._log(f"Step {step_number} started", run_id=run_id, session_id=session_id)
                new_messages = await self._step(
                    messages,
                    python_executor=python_executor,
                    session_id=session_id,
                    run_id=run_id,
                    event_bus=event_bus,
                )
                messages.extend(new_messages)
                self._log(f"Step {step_number} completed", run_id=run_id, session_id=session_id)
                if messages[-1].role == "assistant":
                    break
            else:
                # Final step
                new_messages = await self._handle_final_message(
                    messages, session_id=session_id, run_id=run_id, event_bus=event_bus
                )
                messages.extend(new_messages)
                self._log("Final step completed", run_id=run_id, session_id=session_id)

        except Exception as e:
            self._log(
                f"Agent {self.name} failed with error: {e}",
                run_id=run_id,
                session_id=session_id,
                level=logging.ERROR,
            )
            raise e
        finally:
            # Cleanup
            if python_executor:
                await python_executor.cleanup()
            await self._publish_event(event_bus, session_id, EventType.AGENT_END)

        self._log(f"Agent {self.name} completed successfully", run_id=run_id, session_id=session_id)
        return str(messages[-1].content)

    async def _get_tools(
        self,
        server_host: Optional[str],
        server_port: Optional[int],
    ) -> List[Tool]:
        tools = []
        fetched_tool_names = []
        if server_host and server_port:
            server_url = f"{server_host}:{server_port}"
            tools = await fetch_tools(server_url)
            for tool in tools:
                if tool.description:
                    tool.description = dedent(tool.description).strip()
            tools = [tool for tool in tools if tool.name in self.tool_names]
            fetched_tool_names = [tool.name for tool in tools]

        for tool_name in self.tool_names:
            assert (
                tool_name in fetched_tool_names
            ), f"Tool {tool_name} not found in {fetched_tool_names}"
        return tools

    async def _step(
        self,
        messages: ChatMessages,
        python_executor: PythonExecutor,
        session_id: str,
        run_id: str,
        event_bus: AgentEventBus | None = None,
    ) -> ChatMessages:
        self._log(
            f"Step inputs: {messages}", run_id=run_id, session_id=session_id, level=logging.DEBUG
        )
        output_stream = self.llm.astream(messages, stop=self.prompts.stop_sequences)

        output_text = ""
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
            and not output_text.strip().endswith(self.prompts.end_code_sequence)
        ):
            chunk = self.prompts.end_code_sequence + "\n"
            output_text += chunk

        self._log(
            f"Step output: {output_text}", run_id=run_id, session_id=session_id, level=logging.DEBUG
        )
        code_action = extract_code_from_text(output_text)
        new_messages = []
        if code_action is None:
            new_messages.append(ChatMessage(role="assistant", content=output_text))
            return new_messages

        self._log(
            f"Code action: {code_action}", run_id=run_id, session_id=session_id, level=logging.DEBUG
        )
        tool_call_message = ChatMessage(
            role="assistant",
            content=output_text,
        )
        await self._publish_event(event_bus, session_id, EventType.TOOL_CALL, code_action)
        new_messages.append(tool_call_message)

        # Execute code
        try:
            code_result = await python_executor.invoke(code_action)
            self._log(
                f"Code result: {code_result}",
                run_id=run_id,
                session_id=session_id,
                level=logging.DEBUG,
            )
            code_result_message: ChatMessage = code_result.to_message()
            assert isinstance(code_result_message.content, list)
            new_messages.append(code_result_message)
            tool_output: str = str(code_result_message.content[0]["text"]) + "\n"
            await self._publish_event(event_bus, session_id, EventType.TOOL_RESPONSE, tool_output)
        except Exception as e:
            new_messages.append(ChatMessage(role="user", content=f"Error: {e}"))
            self._log(f"Code error: {e}", run_id=run_id, session_id=session_id, level=logging.DEBUG)
            await self._publish_event(
                event_bus, session_id, EventType.TOOL_RESPONSE, f"Error: {e}\n"
            )
        return new_messages

    async def _handle_final_message(
        self,
        messages: ChatMessages,
        session_id: str,
        run_id: str,
        event_bus: AgentEventBus | None = None,
    ) -> ChatMessages:
        prompt: str = self.prompts.final.render()
        final_message = ChatMessage(role="user", content=prompt)
        input_messages = messages + [final_message]
        self._log(
            f"Final input messages: {input_messages}",
            run_id=run_id,
            session_id=session_id,
            level=logging.DEBUG,
        )

        output_stream = self.llm.astream(input_messages, stop=self.prompts.stop_sequences)
        output_text = ""
        async for event in output_stream:
            if isinstance(event.content, str):
                chunk = event.content
            elif isinstance(event.content, list):
                chunk = "\n".join([str(item) for item in event.content])
            output_text += chunk
            await self._publish_event(event_bus, session_id, EventType.OUTPUT, chunk)

        self._log(
            f"Final message: {final_message}",
            run_id=run_id,
            session_id=session_id,
            level=logging.DEBUG,
        )

        return [ChatMessage(role="assistant", content=output_text)]

    async def _run_planning_step(
        self,
        messages: ChatMessages,
        tools: List[Tool],
        session_id: str,
        event_bus: AgentEventBus | None = None,
    ) -> ChatMessages:
        assert self.prompts.plan is not None, "Planning prompt is not set, but planning is enabled"
        assert (
            self.prompts.plan_prefix is not None
        ), "Plan prefix is not set, but planning is enabled"
        assert (
            self.prompts.plan_suffix is not None
        ), "Plan suffix is not set, but planning is enabled"

        conversation = "\n\n".join([f"{m.role}: {m.content}" for m in messages[1:]])
        planning_prompt = self.prompts.plan.render(conversation=conversation, tools=tools)
        input_messages = [ChatMessage(role="user", content=planning_prompt)]

        output_stream = self.llm.astream(input_messages, stop=[self.prompts.end_plan_sequence])

        plan_prefix = self.prompts.plan_prefix.render().strip() + "\n\n"
        await self._publish_event(event_bus, session_id, EventType.OUTPUT, plan_prefix)
        output_text = plan_prefix

        async for event in output_stream:
            assert isinstance(event.content, str)
            chunk = event.content
            output_text += chunk
            await self._publish_event(event_bus, session_id, EventType.OUTPUT, chunk)

        plan_suffix = "\n\n" + self.prompts.plan_suffix.render().strip() + "\n\n"
        await self._publish_event(event_bus, session_id, EventType.OUTPUT, plan_suffix)
        output_text += plan_suffix

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

    def _log(self, message: str, run_id: str, session_id: str, level: int = logging.INFO) -> None:
        message = f"| {session_id:<8} | {run_id:<8} | {message}"
        self.logger.log(level, message)
