from typing import List

import fire  # type: ignore

from codearkt.event_bus import EventType
from codearkt.llm import ChatMessage
from codearkt.util import get_unique_id
from codearkt.client import query_agent, stop_agent


def main() -> None:
    real_messages: List[ChatMessage] = []
    session_id: str | None = None
    agent_names: List[str] = []
    try:
        while True:
            message = input("You: ")
            if message == "exit":
                break
            session_id = session_id or get_unique_id()
            real_messages.append(ChatMessage(role="user", content=message))
            events = query_agent(real_messages, session_id=session_id)
            for event in events:
                is_root_agent = len(agent_names) == 1
                if event.event_type == EventType.TOOL_RESPONSE:
                    print("Tool Response:\n", event.content)
                elif event.event_type == EventType.AGENT_START:
                    print(f"\n**Starting {event.agent_name} agent...**\n\n")
                    agent_names.append(event.agent_name)
                elif event.event_type == EventType.AGENT_END:
                    print(f"\n**Agent {event.agent_name} completed the task!**\n\n")
                    agent_names.pop()
                elif event.event_type == EventType.OUTPUT:
                    if not event.content:
                        continue
                    print(event.content, end="")
                    if is_root_agent:
                        if real_messages[-1].role == "assistant":
                            assert isinstance(real_messages[-1].content, str)
                            real_messages[-1].content += event.content
                        else:
                            real_messages.append(
                                ChatMessage(role="assistant", content=event.content)
                            )
    except KeyboardInterrupt:
        print("\n\n**Exiting...**\n\n")
        if session_id:
            stop_agent(session_id)
        raise


if __name__ == "__main__":
    fire.Fire(main)
