import re
import socket

from codearkt.util import get_unique_id, find_free_port, truncate_content, is_correct_json


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


def test_truncate_content_short() -> None:
    s = "abc"
    assert truncate_content(s, max_length=10) == "abc"


def test_truncate_content_long() -> None:
    s = "a" * 50
    out = truncate_content(s, max_length=10)
    assert len(out) > 10
    assert out.startswith("aaaaa") and out.endswith("aaaaa")
    assert "truncated" in out


def test_is_correct_json() -> None:
    assert is_correct_json('{"a":1}')
    assert not is_correct_json("not json")
