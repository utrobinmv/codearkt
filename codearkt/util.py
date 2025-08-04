import uuid
import socket
from typing import Optional


def get_unique_id() -> str:
    return str(uuid.uuid4())[:8]


def find_free_port() -> Optional[int]:
    for port in range(5000, 6001):
        try:
            with socket.socket() as s:
                s.bind(("", port))
                return port
        except Exception:
            continue
    return None
