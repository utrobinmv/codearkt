import json
from typing import Any, Callable, Mapping, Tuple, Dict, Collection, Optional, List
from inspect import signature
import logging

from opentelemetry import trace as trace_api
from opentelemetry import context as context_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from openinference.instrumentation import get_attributes_from_context, safe_json_dumps
from openinference.instrumentation import (
    OITracer,
    TraceConfig,
)
from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
    SpanAttributes,
    MessageAttributes,
)
from openinference.instrumentation.openai import OpenAIInstrumentor

from wrapt import wrap_function_wrapper  # type: ignore

from codearkt.codeact import CodeActAgent
from codearkt.python_executor import PythonExecutor
from codearkt.llm import ChatMessage

logger = logging.getLogger(__name__)

INPUT_VALUE = SpanAttributes.INPUT_VALUE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
SESSION_ID = SpanAttributes.SESSION_ID
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
CHAIN = OpenInferenceSpanKindValues.CHAIN.value
AGENT = OpenInferenceSpanKindValues.AGENT.value
TOOL = OpenInferenceSpanKindValues.TOOL.value
LLM = OpenInferenceSpanKindValues.LLM.value


def _bind_arguments(method: Callable[..., Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
    method_signature = signature(method)
    bound_args = method_signature.bind(*args, **kwargs)
    bound_args.apply_defaults()
    return bound_args.arguments


def _strip_method_args(arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in arguments.items() if key not in ("self", "cls")}


def _get_input_value(method: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    arguments = _bind_arguments(method, *args, **kwargs)
    arguments = _strip_method_args(arguments)
    return safe_json_dumps(arguments)


def _get_input_messages(input_value: str) -> List[ChatMessage]:
    input_messages_str: List[str] = json.loads(input_value)["messages"]
    return [ChatMessage.model_validate_json(message_str) for message_str in input_messages_str]


def _get_input_message_attributes(input_value: str) -> Dict[str, Any]:
    input_messages = _get_input_messages(input_value)
    message_attributes = {}
    for idx, message in enumerate(input_messages):
        message_attributes[f"{LLM_INPUT_MESSAGES}.{idx}.{MESSAGE_ROLE}"] = message.role
        message_attributes[f"{LLM_INPUT_MESSAGES}.{idx}.{MESSAGE_CONTENT}"] = str(message.content)
    return message_attributes


def _get_output_message_attributes(messages: List[ChatMessage]) -> Dict[str, Any]:
    message_attributes = {}
    for idx, message in enumerate(messages):
        message_attributes[f"{LLM_OUTPUT_MESSAGES}.{idx}.{MESSAGE_ROLE}"] = message.role
        message_attributes[f"{LLM_OUTPUT_MESSAGES}.{idx}.{MESSAGE_CONTENT}"] = str(message.content)
    return message_attributes


class _AinvokeWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        bound_args = _bind_arguments(wrapped, *args, **kwargs)
        session_id: Optional[str] = bound_args.get("session_id")
        span_name = f"Agent: {instance.name}"

        token: Optional[object] = None
        if session_id is not None:
            token = context_api.attach(context_api.set_value(SESSION_ID, session_id))

        try:
            input_value = _get_input_value(wrapped, *args, **kwargs)
            input_messages = _get_input_messages(input_value)
            conversation = "\n\n".join(
                [f"{message.role}: {message.content}" for message in input_messages]
            )
            message_attributes = _get_input_message_attributes(input_value)

            with self._tracer.start_as_current_span(
                span_name,
                attributes={
                    OPENINFERENCE_SPAN_KIND: AGENT,
                    INPUT_VALUE: conversation,
                    **message_attributes,
                    **({SESSION_ID: session_id} if session_id is not None else {}),
                    **dict(get_attributes_from_context()),
                },
            ) as span:
                result = await wrapped(*args, **kwargs)
                span.set_status(trace_api.StatusCode.OK)
                span.set_attribute(OUTPUT_VALUE, result)
                return result
        except Exception as e:
            raise e
        finally:
            if token is not None:
                context_api.detach(token)  # type: ignore[arg-type]


class _StepWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        bound_args = _bind_arguments(wrapped, *args, **kwargs)
        session_id: Optional[str] = bound_args.get("session_id")

        input_value = _get_input_value(wrapped, *args, **kwargs)
        message_attributes = _get_input_message_attributes(input_value)

        span_name = "Step"
        with self._tracer.start_as_current_span(
            span_name,
            attributes={
                OPENINFERENCE_SPAN_KIND: LLM,
                **message_attributes,
                **({SESSION_ID: session_id} if session_id is not None else {}),
                **dict(get_attributes_from_context()),
            },
        ) as span:
            result: List[ChatMessage] = await wrapped(*args, **kwargs)
            conversation = "\n\n".join([f"{message.role}: {message.content}" for message in result])
            span.set_status(trace_api.StatusCode.OK)
            span.set_attribute(OUTPUT_VALUE, conversation)
            output_message_attributes = _get_output_message_attributes(result)
            span.set_attributes(output_message_attributes)
            return result


class _ToolWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        span_name = "PythonExecutor"

        input_value = _get_input_value(wrapped, *args, **kwargs)
        session_id: Optional[str] = getattr(instance, "session_id", None)

        with self._tracer.start_as_current_span(
            span_name,
            attributes={
                OPENINFERENCE_SPAN_KIND: TOOL,
                INPUT_VALUE: json.loads(input_value)["code"],
                **({SESSION_ID: session_id} if session_id is not None else {}),
                **dict(get_attributes_from_context()),
            },
        ) as span:
            result = await wrapped(*args, **kwargs)

            try:
                span.set_attribute(OUTPUT_VALUE, safe_json_dumps(result))
            except Exception:
                span.set_attribute(OUTPUT_VALUE, str(result))

            span.set_status(trace_api.StatusCode.OK)
            return result


class CodeActInstrumentor(BaseInstrumentor):  # type: ignore
    __slots__ = (
        "_original_step_method",
        "_original_ainvoke_method",
        "_original_final_method",
        "_original_tool_invoke_method",
        "_tracer",
    )

    _original_step_method: Optional[Callable[..., Any]]
    _original_ainvoke_method: Optional[Callable[..., Any]]
    _original_final_method: Optional[Callable[..., Any]]
    _original_tool_invoke_method: Optional[Callable[..., Any]]
    _tracer: OITracer

    def instrumentation_dependencies(self) -> Collection[str]:
        return ["codearkt"]

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        if not (config := kwargs.get("config")):
            config = TraceConfig()
        assert isinstance(config, TraceConfig)
        self._tracer = OITracer(
            trace_api.get_tracer(__name__, "0.1.0", tracer_provider),
            config=config,
        )

        try:
            OpenAIInstrumentor().instrument(
                tracer_provider=tracer_provider,
                config=config,
            )
        except Exception as exc:
            logger.debug("Failed to instrument OpenAI: %s", exc)

        self._original_step_method = getattr(CodeActAgent, "_step", None)
        step_wrapper = _StepWrapper(tracer=self._tracer)
        wrap_function_wrapper(
            module="codearkt.codeact",
            name="CodeActAgent._step",
            wrapper=step_wrapper,
        )

        self._original_final_method = getattr(CodeActAgent, "_handle_final_message", None)
        handle_final_message_wrapper = _StepWrapper(tracer=self._tracer)
        wrap_function_wrapper(
            module="codearkt.codeact",
            name="CodeActAgent._handle_final_message",
            wrapper=handle_final_message_wrapper,
        )

        self._original_ainvoke_method = getattr(CodeActAgent, "ainvoke", None)
        ainvoke_wrapper = _AinvokeWrapper(tracer=self._tracer)
        wrap_function_wrapper(
            module="codearkt.codeact",
            name="CodeActAgent.ainvoke",
            wrapper=ainvoke_wrapper,
        )

        self._original_tool_invoke_method = getattr(PythonExecutor, "invoke", None)
        tool_wrapper = _ToolWrapper(tracer=self._tracer)
        wrap_function_wrapper(
            module="codearkt.python_executor",
            name="PythonExecutor.invoke",
            wrapper=tool_wrapper,
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        if self._original_step_method is not None:
            setattr(CodeActAgent, "_step", self._original_step_method)
            self._original_step_method = None
        if self._original_ainvoke_method is not None:
            setattr(CodeActAgent, "ainvoke", self._original_ainvoke_method)
            self._original_ainvoke_method = None
        if self._original_final_method is not None:
            setattr(CodeActAgent, "_handle_final_message", self._original_final_method)
            self._original_final_method = None
        if self._original_tool_invoke_method is not None:
            setattr(PythonExecutor, "invoke", self._original_tool_invoke_method)
            self._original_tool_invoke_method = None
