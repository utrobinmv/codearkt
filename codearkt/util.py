import uuid
import json
import socket
import random
from typing import Optional

MAX_LENGTH_TRUNCATE_CONTENT: int = 20000


def get_unique_id() -> str:
    return str(uuid.uuid4())[:8]


def find_free_port() -> Optional[int]:
    ports = list(range(5000, 6001))
    random.shuffle(ports)
    for port in ports:
        try:
            with socket.socket() as s:
                s.bind(("", port))
                return port
        except Exception:
            continue
    return None


def truncate_content(content: str, max_length: int = MAX_LENGTH_TRUNCATE_CONTENT) -> str:
    if len(content) <= max_length:
        return content
    else:
        return (
            content[: max_length // 2]
            + f"\n..._This content has been truncated to stay below {max_length} characters_...\n"
            + content[-max_length // 2 :]
        )


def is_correct_json(content: str) -> bool:
    try:
        json.loads(content)
        return True
    except json.JSONDecodeError:
        return False
