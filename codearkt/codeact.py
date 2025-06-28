import re
import uuid
from dataclasses import dataclass
from typing import List, Self, Dict, Any

import yaml

from mcp import Tool
from jinja2 import Template

from codearkt.python_executor import PythonExecutor
from codearkt.mcp_client import fetch_tools
from codearkt.llm import LLM, ChatMessages, ChatMessage, FunctionCall, ToolCall


STOP_SEQUENCES = ["<end_code>", "Observation:", "Calling tools:"]


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
    def load(cls, path: str) -> Self:
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
        llm: LLM,
        prompts: Prompts,
        max_iterations: int = 10,
        mcp_url: str = "http://localhost:5055/mcp",
    ) -> None:
        self.llm: LLM = llm
        self.prompts: Prompts = prompts
        self.max_iterations = max_iterations
        self.python_executor = PythonExecutor()
        self.mcp_url = mcp_url

        self.messages: List[ChatMessage] = []
        self.step_number = 0

    async def ainvoke(self, messages: ChatMessages) -> None:
        tools = await fetch_tools(self.mcp_url)
        self.prompts.format(tools=tools)

        messages = fix_code_actions(messages)
        self.messages = [ChatMessage(role="system", content=self.prompts.system)] + messages

        for step_number in range(self.max_iterations):
            self.step_number = step_number
            await self._step()
            if self.messages[-1].role == "assistant":
                break
        else:
            self.step_number += 1
            await self._handle_final_message()

    async def _step(self) -> None:
        output_text = ""
        output_stream = self.llm.astream(self.messages, stop=STOP_SEQUENCES)
        tool_call_id = f"toolu_{str(uuid.uuid4())[:8]}"
        async for event in output_stream:
            if isinstance(event.content, str):
                output_text += event.content
            elif isinstance(event.content, list):
                output_text += "\n".join([str(item) for item in event.content])

        if (
            output_text
            and output_text.strip().endswith("```")
            and not output_text.strip().endswith("<end_code>")
        ):
            output_text += "<end_code>\n"

        code_action = extract_code_from_text(output_text)
        if code_action is None:
            self.messages.append(ChatMessage(role="assistant", content=output_text))
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
        self.messages.append(tool_call_message)
        try:
            output, execution_logs = self.python_executor.invoke(code_action)
            observation = "Execution logs:\n" + execution_logs
            if output:
                output = str(output)
                observation += "\n\nLast output from code snippet:\n" + output
        except Exception as e:
            observation = f"Error: {e}"
        tool_message = ChatMessage(role="tool", content=observation, tool_call_id=tool_call_id)
        self.messages.append(tool_message)

    async def _handle_final_message(self) -> None:
        prompt = self.prompts.final
        final_message = ChatMessage(role="user", content=prompt)
        self.messages.append(final_message)

        output_stream = self.llm.astream(self.messages, stop=STOP_SEQUENCES)
        output_text = ""
        async for event in output_stream:
            if isinstance(event.content, str):
                output_text += event.content
        self.messages.append(ChatMessage(role="assistant", content=output_text))


if __name__ == "__main__":
    import asyncio
    import os
    from dotenv import load_dotenv

    load_dotenv()

    llm = LLM(
        model_name="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key=os.getenv("OPENAI_API_KEY", ""),
    )
    prompts = Prompts.load("codearkt/prompts/codeact.yaml")
    agent = CodeActAgent(llm, prompts)
    asyncio.run(
        agent.ainvoke(
            [
                ChatMessage(
                    role="user", content="Print the abstract of the PingPong paper by Ilya Gusev"
                )
            ]
        )
    )
