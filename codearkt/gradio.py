import requests
from typing import Iterator, List, Tuple, Dict, Any

import gradio as gr

from codearkt.event_bus import AgentEvent, EventType


HEADERS = {"Content-Type": "application/json", "Accept": "text/event-stream"}
CODE_TITLE = "Code execution result"


def query_manager_agent(
    query: str,
    base_url: str = "http://localhost:5055",
) -> Iterator[AgentEvent]:
    url = f"{base_url}/agents/manager"
    payload = {"query": query, "stream": True}

    response = requests.post(url, json=payload, headers=HEADERS, stream=True, timeout=30)
    response.raise_for_status()
    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
        if chunk:
            yield AgentEvent.model_validate_json(chunk)


def user(user_message: str, history: List[gr.ChatMessage]) -> Tuple[str, List[gr.ChatMessage]]:
    new_history = history + [gr.ChatMessage(role="user", content=user_message)]
    return "", new_history


def bot(history: List[Dict[str, Any]]) -> Iterator[List[Dict[str, Any]]]:
    last_user_message = history[-1]

    history.append({"role": "assistant", "content": ""})
    for event in query_manager_agent(last_user_message["content"]):
        if event.event_type == EventType.TOOL_RESPONSE:
            assert event.content is not None
            history.append(
                {
                    "role": "assistant",
                    "content": event.content,
                    "metadata": {"title": CODE_TITLE, "status": "done"},
                }
            )
            continue

        prev_message = history[-1]
        prev_message_meta: Dict[str, Any] = prev_message.get("metadata", {})
        prev_message_title = prev_message_meta.get("title")
        if prev_message_title == CODE_TITLE:
            # Start new assistant message
            assert event.content is not None
            history.append({"role": "assistant", "content": event.content})
            continue

        if event.event_type == EventType.AGENT_START:
            history.append(
                {"role": "assistant", "content": f'**Starting "{event.agent_name}" agent...**\n'}
            )
            continue

        if event.event_type == EventType.AGENT_END:
            history.append(
                {
                    "role": "assistant",
                    "content": f"**Agent {event.agent_name} completed the task!**\n",
                }
            )
            continue

        assert event.content is not None
        history[-1]["content"] += event.content
        yield history


class GradioUI:
    def create_app(self) -> Any:
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            chatbot = gr.Chatbot(type="messages")
            with gr.Row():
                with gr.Column():
                    msg = gr.Textbox(
                        label="Send message",
                        placeholder="Send message",
                        show_label=False,
                    )
                with gr.Column():
                    with gr.Row():
                        submit = gr.Button("Send")
                        stop = gr.Button("Stop")
            submit_event = msg.submit(
                fn=user,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot],
                queue=False,
            ).success(
                fn=bot,
                inputs=[
                    chatbot,
                ],
                outputs=chatbot,
                queue=True,
            )
            submit_click_event = submit.click(
                fn=user,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot],
                queue=False,
            ).success(
                fn=bot,
                inputs=[
                    chatbot,
                ],
                outputs=chatbot,
                queue=True,
            )

            stop.click(
                fn=None,
                inputs=None,
                outputs=None,
                cancels=[submit_event, submit_click_event],
                queue=False,
            )
        return demo

    def run(self) -> None:
        self.create_app().launch(show_error=True)


if __name__ == "__main__":
    GradioUI().run()
