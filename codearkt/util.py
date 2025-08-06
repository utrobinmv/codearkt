import uuid
import socket
import random
from typing import Optional


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
