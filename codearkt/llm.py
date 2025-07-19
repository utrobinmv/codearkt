import os
from typing import Dict, Any, List, cast, AsyncGenerator, Optional

from dotenv import load_dotenv
from pydantic import BaseModel
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, ChoiceDelta

load_dotenv()
BASE_URL = os.getenv("BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = os.getenv("OPENROUTER_API_KEY", "")


class FunctionCall(BaseModel):  # type: ignore
    name: str
    arguments: str


class ToolCall(BaseModel):  # type: ignore
    id: str
    type: str = "function"
    function: FunctionCall


class ChatMessage(BaseModel):  # type: ignore
    role: str
    content: str | List[Dict[str, Any]]
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

    def __str__(self) -> str:
        dump: str = self.model_dump_json()
        return dump


ChatMessages = List[ChatMessage]


class LLM:
    def __init__(
        self,
        model_name: str,
        base_url: str = BASE_URL,
        api_key: str = API_KEY,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_tokens: int = 8192,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self._base_url = base_url
        self._api_key = api_key
        self.params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stop": stop,
        }
        for k, v in kwargs.items():
            if k not in self.params:
                self.params[k] = v

    async def astream(
        self, messages: ChatMessages, **kwargs: Any
    ) -> AsyncGenerator[ChoiceDelta, None]:
        casted_messages = [
            cast(ChatCompletionMessageParam, message.model_dump(exclude_none=True))
            for message in messages
        ]

        api_params = {**self.params, **kwargs}

        async with AsyncOpenAI(base_url=self._base_url, api_key=self._api_key) as api:
            stream: AsyncStream[ChatCompletionChunk] = await api.chat.completions.create(
                model=self.model_name,
                messages=casted_messages,
                stream=True,
                tool_choice="none",
                **api_params,
            )
            async for event in stream:
                event_typed: ChatCompletionChunk = event
                delta = event_typed.choices[0].delta
                yield delta
