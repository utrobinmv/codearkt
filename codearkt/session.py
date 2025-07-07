# orchestrator.py
import docker
import textwrap
import time
import httpx
import asyncio
import atexit
import signal
import sys
from typing import Literal, Tuple


IMAGE = "phoenix120/codearkt_http"
MEM_LIMIT = "512m"
CPU_QUOTA = 50000
CPU_PERIOD = 100000
EXEC_TIMEOUT = 30  # seconds per code block
SESSION_TTL = 60 * 60  # kill container after 60 min


class Session:
    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
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
        self._shutdown_called = False

        atexit.register(self._cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def __del__(self):
        self._cleanup()

    def _signal_handler(self, signum, frame):
        print(f"Received signal {signum}, shutting down container...")
        self.shutdown()
        sys.exit(0)

    def _cleanup(self):
        if not self._shutdown_called:
            try:
                if hasattr(self, "_container"):
                    self._container.kill()
                    print(f"Container {self._container.short_id} stopped and removed")
            except Exception as e:
                print(f"Error during cleanup: {e}")

    # ---------------------------------------------------------------------
    def _get_url(self) -> str:
        self._container.reload()
        ports = self._container.attrs["NetworkSettings"]["Ports"]
        mapping = ports["8000/tcp"][0]
        return f"http://localhost:{mapping['HostPort']}"

    async def _wait_for_ready(self, max_wait: int = 60) -> None:
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                # Test with a simple piece of code
                test_payload = {"code": "print('ready')"}
                async with httpx.AsyncClient() as client:
                    resp = await client.post(f"{self._url}/exec", json=test_payload, timeout=2)
                    if resp.status_code == 200:
                        print(resp.json(), resp.status_code)
                        return
                    else:
                        print(resp.text)
            except (httpx.RequestError, httpx.TimeoutException):
                print("Container not ready yet")
                pass
            await asyncio.sleep(0.5)
        raise RuntimeError("Container failed to become ready within timeout")

    async def run(self, code: str) -> Tuple[str, str]:
        await self._wait_for_ready()
        if time.monotonic() - self._start > SESSION_TTL:
            raise RuntimeError("Session TTL exceeded; create a new Session.")
        payload = {"code": textwrap.dedent(code), "session_id": self._session_id}
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{self._url}/exec", json=payload, timeout=EXEC_TIMEOUT)
            resp.raise_for_status()
            out = resp.json()
            output: str = out.get("stdout", "")
            if out.get("error"):
                output += "\n" + out["error"]
            return "", output

    def get_logs(self, tail: Literal["all"] | int = "all") -> str:
        return self._container.logs(tail=tail).decode("utf-8")

    def shutdown(self) -> None:
        if not self._shutdown_called:
            self._shutdown_called = True
            try:
                self._container.kill()  # type: ignore
                print(f"Container {self._container.short_id} shutdown and removed")
            except Exception as e:
                print(f"Error during shutdown: {e}")


if __name__ == "__main__":
    # Example using context manager for automatic cleanup
    async def main():
        sess = Session("testid")
        print(
            await sess.run(
                """
import json
answer = json.loads(arxiv_download(paper_id="2506.15003"))["title"]
print("Answer 1:", answer)
"""
            )
        )

        print(
            await sess.run(
                """
print("Variable still here:", answer[:5], "…")
"""
            )
        )

    asyncio.run(main())
