import textwrap
import time
import httpx
import asyncio
import atexit
import signal
from typing import Optional, Any, List

import docker
from docker.models.containers import Container


IMAGE = "phoenix120/codearkt_http"
MEM_LIMIT = "512m"
CPU_QUOTA = 50000
CPU_PERIOD = 100000
EXEC_TIMEOUT = 30
PIDS_LIMIT = 64
CLIENT = None
NET_NAME = "sandbox_net"


class PythonExecutor:
    def __init__(
        self,
        session_id: str,
        tool_names: List[str],
    ) -> None:
        global CLIENT
        if CLIENT is None:
            CLIENT = docker.from_env()

        try:
            net = CLIENT.networks.get(NET_NAME)
        except docker.errors.NotFound:
            net = CLIENT.networks.create(
                NET_NAME,
                driver="bridge",
                internal=False,
                options={
                    "com.docker.network.bridge.enable_ip_masquerade": "false",
                },
            )

        self.session_id = session_id
        self.tool_names = tool_names

        self.container: Optional[Container] = CLIENT.containers.run(
            IMAGE,
            detach=True,
            auto_remove=True,
            ports={"8000/tcp": None},
            mem_limit=MEM_LIMIT,
            cpu_period=CPU_PERIOD,
            cpu_quota=CPU_QUOTA,
            pids_limit=PIDS_LIMIT,
            cap_drop=["ALL"],
            read_only=True,
            tmpfs={"/tmp": "rw,size=64m", "/run": "rw,size=16m"},
            security_opt=["no-new-privileges"],
            extra_hosts={"host.docker.internal": "host-gateway"},
            sysctls={"net.ipv4.ip_forward": "0"},
            user="1000:1000",
            network=net.name,
            dns=[],
        )
        self.start = time.monotonic()
        self.url = self._get_url()

        atexit.register(self._cleanup)
        signal.signal(signal.SIGTERM, self._cleanup)
        signal.signal(signal.SIGINT, self._cleanup)

    async def invoke(self, code: str) -> str:
        await self._wait_for_ready()
        return await self._call_exec(code)

    async def _call_exec(self, code: str) -> str:
        payload = {
            "code": textwrap.dedent(code),
            "session_id": self.session_id,
            "tool_names": self.tool_names,
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{self.url}/exec", json=payload, timeout=EXEC_TIMEOUT)
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
        if self.container:
            try:
                self.container.stop()
                self.container.remove(force=True)
            except Exception:
                pass
            finally:
                self.container = None
        if signum == signal.SIGINT:
            raise KeyboardInterrupt()

    def _get_url(self) -> str:
        assert self.container is not None
        self.container.reload()
        ports = self.container.attrs["NetworkSettings"]["Ports"]
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
                pass
            await asyncio.sleep(0.5)
        raise RuntimeError("Container failed to become ready within timeout")
