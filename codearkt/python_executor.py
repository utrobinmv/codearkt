import textwrap
import time
import httpx
import asyncio
import atexit
import signal
from typing import Optional, Any

import docker
from docker.models.containers import Container


IMAGE = "phoenix120/codearkt_http"
MEM_LIMIT = "512m"
CPU_QUOTA = 50000
CPU_PERIOD = 100000
EXEC_TIMEOUT = 30
CLIENT = None


class PythonExecutor:
    def __init__(self, session_id: str) -> None:
        global CLIENT
        if CLIENT is None:
            CLIENT = docker.from_env()

        self._session_id = session_id
        self._container: Optional[Container] = CLIENT.containers.run(
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

        atexit.register(self._cleanup)
        signal.signal(signal.SIGTERM, self._cleanup)
        signal.signal(signal.SIGINT, self._cleanup)

    async def invoke(self, code: str) -> str:
        await self._wait_for_ready()
        return await self._call_exec(code)

    async def _call_exec(self, code: str) -> str:
        payload = {"code": textwrap.dedent(code), "session_id": self._session_id}

        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{self._url}/exec", json=payload, timeout=EXEC_TIMEOUT)
            resp.raise_for_status()
            out = resp.json()

            output: str = ""
            if out.get("stdout"):
                output += out["stdout"] + "\n\n"
            if out.get("error"):
                output += "Error: " + out["error"]
            output = output.strip()
            return output

    def _cleanup(self, signum: Optional[Any] = None, frame: Optional[Any] = None) -> None:
        if self._container:
            try:
                self._container.remove(force=True)
                self._container = None
            except Exception:
                pass
        if signum == signal.SIGINT:
            raise KeyboardInterrupt()

    def _get_url(self) -> str:
        assert self._container is not None
        self._container.reload()
        ports = self._container.attrs["NetworkSettings"]["Ports"]
        mapping = ports["8000/tcp"][0]
        return f"http://localhost:{mapping['HostPort']}"

    async def _wait_for_ready(self, max_wait: int = 60) -> None:
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                result = await self._call_exec("print('ready')")
                assert result.strip() == "ready"
                return
            except (httpx.RequestError, httpx.TimeoutException, AssertionError):
                print("Container not ready yet")
                pass
            await asyncio.sleep(0.5)
        raise RuntimeError("Container failed to become ready within timeout")
