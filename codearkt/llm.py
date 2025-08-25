import os
import copy
import logging
import asyncio
from contextlib import suppress
from typing import Dict, Any, List, cast, AsyncGenerator, Optional

from httpx import Timeout
from tiktoken import encoding_for_model
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

load_dotenv()

BASE_URL = os.getenv("BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = os.getenv("OPENROUTER_API_KEY", "")
HTTP_REFERRER = os.getenv("HTTP_REFERRER", "https://github.com/IlyaGusev/codearkt/")
X_TITLE = os.getenv("X_TITLE", "CodeArkt")


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


def count_openai_tokens(messages: ChatMessages) -> int:
    encoding = encoding_for_model("gpt-4o")
    tokens = [encoding.encode(str(message.content)) for message in messages]
    return sum(len(token) for token in tokens) + 3 * len(messages)


class LLM:
    def __init__(
        self,
        model_name: str,
        base_url: str = BASE_URL,
        api_key: str = API_KEY,
        max_history_tokens: int = 200000,
        num_retries: int = 3,
        connect_timeout_sec: float = 10.0,
        read_timeout_sec: float = 600.0,
        write_timeout_sec: float = 600.0,
        event_idle_timeout_sec: float = 60.0,
        pool_timeout_sec: float = 600.0,
        **kwargs: Any,
    ) -> None:
        self._model_name = model_name
        self._base_url = base_url
        self._api_key = api_key
        self._max_history_tokens = max_history_tokens
        self._params: Dict[str, Any] = {}
        self._num_retries = num_retries
        self._logger = logging.getLogger(self.__class__.__name__)
        self._event_idle_timeout_sec = event_idle_timeout_sec
        self._http_timeout = Timeout(
            connect=connect_timeout_sec,
            read=read_timeout_sec,
            write=write_timeout_sec,
            pool=pool_timeout_sec,
        )
        for k, v in kwargs.items():
            self._params[k] = v

    def _trim_messages(self, messages: ChatMessages) -> ChatMessages:
        tokens_count = count_openai_tokens(messages)
        system_message = None
        if messages[0].role == "system":
            system_message = messages[0]
            messages = messages[1:]
        while tokens_count > self._max_history_tokens and len(messages) >= 2:
            tokens_count -= count_openai_tokens(messages[:2])
            messages = messages[2:]
        if system_message:
            messages = [system_message] + messages
        return messages

    async def astream(
        self, messages: ChatMessages, **kwargs: Any
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        messages = copy.deepcopy(messages)
        api_params = {**self._params, **kwargs}

        if "gpt-5" in self._model_name:
            if messages[0].role == "system":
                messages[0].role = "developer"
            api_params.pop("stop", None)
            if "max_tokens" in api_params:
                api_params["max_completion_tokens"] = api_params.pop("max_tokens")

        messages = self._trim_messages(messages)
        casted_messages = [
            cast(ChatCompletionMessageParam, message.model_dump(exclude_none=True))
            for message in messages
        ]

        async with AsyncOpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            timeout=self._http_timeout,
            max_retries=self._num_retries,
        ) as api:
            stream: AsyncStream[ChatCompletionChunk] = await api.chat.completions.create(
                model=self._model_name,
                messages=casted_messages,
                stream=True,
                extra_headers={
                    "HTTP-Referer": HTTP_REFERRER,
                    "X-Title": X_TITLE,
                },
                timeout=self._http_timeout,
                **api_params,
            )
            stream_iter = stream.__aiter__()
            try:
                while True:
                    try:
                        event = await asyncio.wait_for(
                            stream_iter.__anext__(), timeout=self._event_idle_timeout_sec
                        )
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError as e:
                        raise TimeoutError(
                            f"LLM stream idle for {self._event_idle_timeout_sec} seconds"
                        ) from e
                    event_typed: ChatCompletionChunk = event
                    yield event_typed
            except asyncio.CancelledError:
                with suppress(Exception):
                    await stream.close()
                raise
            finally:
                with suppress(Exception):
                    await stream.close()
