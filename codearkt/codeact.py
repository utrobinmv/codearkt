import re
import copy
import logging
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
from codearkt.util import get_unique_id


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
    plan: Optional[Template] = None
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
        verosity_level: int = logging.ERROR,
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
        self.verosity_level = verosity_level
        self.planning_interval = planning_interval
        self.server_url = server_url
        self.managed_agents: Optional[List[Self]] = managed_agents
        if self.managed_agents:
            for agent in self.managed_agents:
                agent_tool_name = "agent__" + agent.name
                if agent_tool_name not in self.tool_names:
                    self.tool_names.append(agent_tool_name)

        self.logger = logging.getLogger(self.__class__.__name__ + ":" + self.name)
        self.logger.setLevel(self.verosity_level)
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
    ) -> str:
        messages = copy.deepcopy(messages)

        # Register agent start
        run_id = get_unique_id()
        await self._publish_event(event_bus, session_id, EventType.AGENT_START)
        self._log(f"Starting agent {self.name}", run_id=run_id, session_id=session_id)

        try:
            # Initialize Python interpreter
            python_executor = PythonExecutor(
                session_id=session_id, tool_names=self.tool_names, interpreter_id=run_id
            )
            self._log("Python interpreter started", run_id=run_id, session_id=session_id)

            # Check tools
            tools = []
            if self.server_url:
                tools = await fetch_tools(self.server_url)
                tools = [tool for tool in tools if tool.name in self.tool_names]
                for tool_name in self.tool_names:
                    assert tool_name in [tool.name for tool in tools], f"Tool {tool_name} not found"
            system_prompt = self.prompts.system.render(tools=tools)
            self._log(f"Available tools: {self.tool_names}", run_id=run_id, session_id=session_id)

            # Form input messages
            messages = fix_code_actions(messages)
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
            await python_executor.cleanup()
            await self._publish_event(event_bus, session_id, EventType.AGENT_END)

        self._log(f"Agent {self.name} completed successfully", run_id=run_id, session_id=session_id)
        return str(messages[-1].content)

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
        output_stream = self.llm.astream(messages, stop=STOP_SEQUENCES)

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
            and not output_text.strip().endswith(END_CODE_SEQUENCE)
        ):
            chunk = END_CODE_SEQUENCE + "\n"
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
        tool_call_id = f"toolu_{get_unique_id()}"
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
            self._log(
                f"Code result: {code_result}",
                run_id=run_id,
                session_id=session_id,
                level=logging.DEBUG,
            )
            code_result_messages = code_result.to_messages(tool_call_id)
            new_messages.extend(code_result_messages)
            tool_output: str = str(code_result_messages[0].content) + "\n"
            await self._publish_event(event_bus, session_id, EventType.TOOL_RESPONSE, tool_output)
        except Exception as e:
            new_messages.append(
                ChatMessage(role="tool", content=f"Error: {e}", tool_call_id=tool_call_id)
            )
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

        output_stream = self.llm.astream(input_messages, stop=STOP_SEQUENCES)
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

        conversation = "\n\n".join([f"{m.role}: {m.content}" for m in messages])
        planning_prompt = self.prompts.plan.render(conversation=conversation, tools=tools)
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

    def _log(self, message: str, run_id: str, session_id: str, level: int = logging.INFO) -> None:
        message = f"| {session_id:<8} | {run_id:<8} | {message}"
        self.logger.log(level, message)
