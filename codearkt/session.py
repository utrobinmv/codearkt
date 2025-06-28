# orchestrator.py
import docker
import textwrap
import time
import requests
from typing import Literal, Tuple


IMAGE = "phoenix120/codearkt_http"
MEM_LIMIT = "512m"
CPU_QUOTA = 50000
CPU_PERIOD = 100000
EXEC_TIMEOUT = 30  # seconds per code block
SESSION_TTL = 60 * 60  # kill container after 60 min


class Session:
    def __init__(self) -> None:
        self._client = docker.from_env()
        self._container = self._client.containers.run(
            IMAGE,
            detach=True,
            auto_remove=True,
            ports={"8000/tcp": None},
            mem_limit=MEM_LIMIT,
            cpu_period=CPU_PERIOD,
            cpu_quota=CPU_QUOTA,
            pids_limit=128,
            security_opt=["no-new-privileges"],
        )
        self._start = time.monotonic()
        self._url = self._get_url()
        self._wait_for_ready()

    # ---------------------------------------------------------------------
    def _get_url(self) -> str:
        self._container.reload()
        ports = self._container.attrs["NetworkSettings"]["Ports"]
        mapping = ports["8000/tcp"][0]
        return f"http://localhost:{mapping['HostPort']}"

    def _wait_for_ready(self, max_wait: int = 30) -> None:
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                # Test with a simple piece of code
                test_payload = {"code": "print('ready')"}
                resp = requests.post(f"{self._url}/exec", json=test_payload, timeout=2)
                if resp.status_code == 200:
                    return
            except (requests.RequestException, requests.ConnectionError):
                pass
            time.sleep(0.5)
        raise RuntimeError("Container failed to become ready within timeout")

    def run(self, code: str) -> Tuple[str, str]:
        if time.monotonic() - self._start > SESSION_TTL:
            raise RuntimeError("Session TTL exceeded; create a new Session.")
        payload = {"code": textwrap.dedent(code)}
        resp = requests.post(f"{self._url}/exec", json=payload, timeout=EXEC_TIMEOUT)
        resp.raise_for_status()
        out = resp.json()
        output: str = out.get("stdout", "")
        if out.get("error"):
            output += "\n" + out["error"]
        return "", output

    def get_logs(self, tail: Literal["all"] | int = "all") -> str:
        return self._container.logs(tail=tail).decode("utf-8")

    def shutdown(self) -> None:
        self._container.kill()  # type: ignore


if __name__ == "__main__":
    sess = Session()
    print(
        sess.run(
            """
import json
answer = json.loads(arxiv_download(paper_id="2506.15003"))["title"]
print("Answer 1:", answer)
"""
        )
    )

    print(
        sess.run(
            """
print("Variable still here:", answer[:5], "…")
"""
        )
    )
