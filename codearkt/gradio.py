import requests
from typing import Iterator, List, Dict, Any
import uuid

import gradio as gr

from codearkt.event_bus import AgentEvent, EventType
from codearkt.llm import ChatMessage
from contextlib import closing


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

    with closing(
        requests.post(url, json=payload, headers=HEADERS, stream=True, timeout=20)
    ) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                yield AgentEvent.model_validate_json(chunk)


def stop_agent(session_id: str) -> None:
    try:
        requests.post(f"{BASE_URL}/agents/cancel", json={"session_id": session_id}, timeout=5)
    except requests.RequestException:
        pass


def bot(
    message: str,
    history: List[Dict[str, Any]],
    session_id: str | None,
) -> Iterator[tuple[List[Dict[str, Any]], str | None]]:
    history.append({"role": "assistant", "content": ""})

    session_id = str(uuid.uuid4())
    events = query_manager_agent([ChatMessage(role="user", content=message)], session_id=session_id)
    for event in events:
        session_id = event.session_id or session_id

        if event.event_type == EventType.TOOL_RESPONSE:
            assert event.content is not None
            history.append(
                {
                    "role": "assistant",
                    "content": event.content,
                    "metadata": {"title": CODE_TITLE, "status": "done"},
                }
            )
            yield history, session_id
            continue

        prev_message = history[-1]
        prev_message_meta: Dict[str, Any] = prev_message.get("metadata", {})
        prev_message_title = prev_message_meta.get("title")

        if prev_message_title == CODE_TITLE:
            # Start new assistant message
            assert event.content is not None
            history.append({"role": "assistant", "content": event.content})
            yield history, session_id
            continue

        if event.event_type == EventType.AGENT_START:
            history.append(
                {
                    "role": "assistant",
                    "content": f'**Starting "{event.agent_name}" agent...**\n',
                }
            )
            yield history, session_id
            continue

        if event.event_type == EventType.AGENT_END:
            history.append(
                {
                    "role": "assistant",
                    "content": f"**Agent {event.agent_name} completed the task!**\n",
                }
            )
            yield history, session_id
            continue

        assert event.content is not None
        history[-1]["content"] += event.content
        yield history, session_id


class GradioUI:
    def create_app(self) -> Any:
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            session_state = gr.State(None)

            chat_iface = gr.ChatInterface(
                bot,
                type="messages",
                additional_inputs=[session_state],
                additional_outputs=[session_state],
                save_history=True,
            )

            def _on_stop(session_id: str | None) -> None:
                if session_id:
                    stop_agent(session_id)

            chat_iface.textbox.stop(_on_stop, inputs=[session_state], outputs=None)
        return demo

    def run(self) -> None:
        self.create_app().launch(show_error=True)


if __name__ == "__main__":
    GradioUI().run()
