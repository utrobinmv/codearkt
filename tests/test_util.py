import re
import socket

from codearkt.util import get_unique_id, find_free_port


def test_get_unique_id_length_and_uniqueness() -> None:
    ids = [get_unique_id() for _ in range(1000)]
    assert all(len(uid) == 8 for uid in ids)
    hex_regex = re.compile(r"^[0-9a-f]{8}$", re.IGNORECASE)
    assert all(hex_regex.match(uid) for uid in ids)
    assert len(ids) == len(set(ids))


def test_find_free_port() -> None:
    port = find_free_port()
    assert port is not None, "No free port returned"
    assert 5000 <= port <= 6000, "Port outside expected range"

    with socket.socket() as s:
        s.bind(("", port))
