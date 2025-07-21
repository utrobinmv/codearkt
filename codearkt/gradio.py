import httpx
from typing import Iterator, List, Dict, Any

import gradio as gr

from codearkt.event_bus import AgentEvent, EventType
from codearkt.llm import ChatMessage, ToolCall, FunctionCall
from codearkt.util import get_unique_id

HEADERS = {"Content-Type": "application/json", "Accept": "text/event-stream"}
CODE_TITLE = "Code execution result"
BASE_URL = "http://localhost:5055"


def query_manager_agent(
    history: List[ChatMessage],
    *,
    session_id: str | None = None,
    base_url: str = BASE_URL,
) -> Iterator[AgentEvent]:
    url = f"{base_url}/agents/manager"
    serialized_history = [m.model_dump() for m in history]
    payload = {"messages": serialized_history, "stream": True}
    if session_id is not None:
        payload["session_id"] = session_id

    timeout = httpx.Timeout(connect=10, pool=None, read=None, write=None)
    with httpx.stream("POST", url, json=payload, headers=HEADERS, timeout=timeout) as response:
        response.raise_for_status()
        for chunk in response.iter_text():
            if chunk:
                yield AgentEvent.model_validate_json(chunk)


def stop_agent(session_id: str) -> None:
    try:
        httpx.post(f"{BASE_URL}/agents/cancel", json={"session_id": session_id}, timeout=5.0)
    except httpx.HTTPError:
        pass


def clean_real_messages(messages: List[ChatMessage]) -> List[ChatMessage]:
    prev_message = None
    for message in messages:
        if message.role == "assistant" and message.tool_calls:
            code = message.tool_calls[0].function.arguments
            if (
                prev_message
                and prev_message.role == "assistant"
                and isinstance(prev_message.content, str)
            ):
                prev_message.content = prev_message.content.replace(code, "")
        prev_message = message
    return messages


def bot(
    message: str,
    history: List[Dict[str, Any]],
    session_id: str | None,
    real_messages: List[ChatMessage],
) -> Iterator[tuple[List[Dict[str, Any]], str | None, List[ChatMessage]]]:
    history = []
    session_id = session_id or get_unique_id()
    real_messages = clean_real_messages(real_messages)
    real_messages.append(ChatMessage(role="user", content=message))
    events = query_manager_agent(real_messages, session_id=session_id)
    agent_names: List[str] = []
    history.append({"role": "assistant", "content": ""})
    for event in events:
        session_id = event.session_id or session_id
        prev_message = history[-1]
        prev_message_meta: Dict[str, Any] = prev_message.get("metadata", {})
        prev_message_title = prev_message_meta.get("title")
        is_root_agent = len(agent_names) == 1

        if event.event_type == EventType.TOOL_RESPONSE:
            assert event.content
            if is_root_agent:
                assert real_messages[-1].tool_calls
                tool_call_id = real_messages[-1].tool_calls[0].id
                assert tool_call_id
                real_messages.append(
                    ChatMessage(role="tool", content=event.content, tool_call_id=tool_call_id)
                )
            history.append(
                {
                    "role": "assistant",
                    "content": event.content,
                    "metadata": {"title": CODE_TITLE, "status": "done"},
                }
            )
        elif event.event_type == EventType.AGENT_START:
            agent_names.append(event.agent_name)
            history.append(
                {
                    "role": "assistant",
                    "content": f'\n**Starting "{event.agent_name}" agent...**\n\n',
                }
            )
        elif event.event_type == EventType.AGENT_END:
            last_agent_name = agent_names.pop()
            assert last_agent_name == event.agent_name
            history.append(
                {
                    "role": "assistant",
                    "content": f"\n**Agent {event.agent_name} completed the task!**\n\n",
                }
            )
        elif event.event_type == EventType.TOOL_CALL:
            assert event.content
            tool_call_id = get_unique_id()
            if is_root_agent:
                real_messages.append(
                    ChatMessage(
                        role="assistant",
                        content="",
                        tool_calls=[
                            ToolCall(
                                id=tool_call_id,
                                function=FunctionCall(
                                    name="python_interpreter", arguments=event.content
                                ),
                            )
                        ],
                    )
                )
        else:
            assert event.event_type == EventType.OUTPUT
            if prev_message_title == CODE_TITLE:
                history.append({"role": "assistant", "content": ""})

            assert event.content is not None
            history[-1]["content"] += event.content

            if is_root_agent:
                if real_messages[-1].role == "assistant" and not real_messages[-1].tool_calls:
                    assert isinstance(real_messages[-1].content, str)
                    real_messages[-1].content += event.content
                else:
                    real_messages.append(ChatMessage(role="assistant", content=event.content))

        yield history, session_id, real_messages


class GradioUI:
    def create_app(self) -> Any:
        with gr.Blocks(theme=gr.themes.Soft(), fill_height=True, fill_width=True) as demo:
            session_id_state = gr.State(None)
            real_messages_state = gr.State([])

            chat_iface = gr.ChatInterface(
                bot,
                type="messages",
                additional_inputs=[session_id_state, real_messages_state],
                additional_outputs=[session_id_state, real_messages_state],
                save_history=True,
            )

            def _on_stop(session_id: str | None) -> str | None:
                if session_id:
                    stop_agent(session_id)
                return session_id

            chat_iface.textbox.stop(_on_stop, inputs=[session_id_state], outputs=[session_id_state])
        return demo

    def run(self) -> None:
        self.create_app().launch(show_error=True)


if __name__ == "__main__":
    GradioUI().run()
