import textwrap
import time
import httpx
import asyncio
import atexit
import signal
import json
from typing import Optional, Any, List, Dict, Sequence

import docker
from docker.models.containers import Container
from docker.client import DockerClient
from pydantic import BaseModel

from codearkt.llm import ChatMessage
from codearkt.tools import fetch_tools
from codearkt.util import get_unique_id


IMAGE: str = "phoenix120/codearkt_http"
MEM_LIMIT: str = "512m"
CPU_QUOTA: int = 50000
CPU_PERIOD: int = 100000
EXEC_TIMEOUT: int = 600
CLEANUP_TIMEOUT: int = 30
PIDS_LIMIT: int = 64
NET_NAME: str = "sandbox_net"
CONTAINER_NAME: str = "codearkt_http"

_CLIENT: Optional[DockerClient] = None
_CONTAINER: Optional[Container] = None


def cleanup_container(signum: Optional[Any] = None, frame: Optional[Any] = None) -> None:
    global _CONTAINER
    if _CONTAINER:
        try:
            _CONTAINER.remove(force=True)
            _CONTAINER = None
        except Exception:
            pass
    if signum == signal.SIGINT:
        raise KeyboardInterrupt()


atexit.register(cleanup_container)
signal.signal(signal.SIGINT, cleanup_container)
signal.signal(signal.SIGTERM, cleanup_container)


MAX_LENGTH_TRUNCATE_CONTENT: int = 20000


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


class ExecResult(BaseModel):  # type: ignore
    stdout: str
    error: str | None = None
    result: Any | None = None

    def to_messages(self, tool_call_id: str) -> List[ChatMessage]:
        image_content: List[Dict[str, Any]] | None = None
        output: str = "Stdout:\n" + self.stdout + "\n\n"
        if self.result is not None:
            try:
                json_result = json.loads(str(self.result))
                if isinstance(json_result, dict) and "image_base64" in json_result:
                    image_base64 = json_result["image_base64"]
                    image_content = [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                        }
                    ]
            except json.JSONDecodeError:
                pass
            if not image_content:
                output += "Last expression:\n" + str(self.result) + "\n\n"
        if self.error:
            output += "Error: " + self.error
        output = output.strip()

        messages = []
        messages.append(ChatMessage(role="tool", content=output, tool_call_id=tool_call_id))
        if image_content:
            messages.append(ChatMessage(role="user", content=image_content))
        return messages


def init_docker() -> docker.DockerClient:
    client = docker.from_env()
    try:
        client.ping()  # type: ignore
    except docker.errors.DockerException as exc:
        raise RuntimeError(
            "Docker daemon is not running or not accessible – skipping PythonExecutor setup."
        ) from exc

    try:
        client.images.get(IMAGE)
    except docker.errors.ImageNotFound:
        try:
            client.images.pull(IMAGE)
        except docker.errors.DockerException as exc:
            raise RuntimeError(
                f"Docker image '{IMAGE}' not found locally and failed to pull automatically."
            ) from exc
    except docker.errors.DockerException as exc:
        raise RuntimeError("Failed to query Docker images – ensure Docker is available.") from exc
    return client


class PythonExecutor:
    def __init__(
        self,
        tool_names: Sequence[str] = tuple(),
        session_id: Optional[str] = None,
        mcp_server_port: int = 5055,
        interpreter_id: Optional[str] = None,
    ) -> None:
        global _CLIENT, _CONTAINER

        if not _CLIENT:
            _CLIENT = init_docker()
        client = _CLIENT

        if _CONTAINER is None:
            try:
                _CONTAINER = _CLIENT.containers.get(CONTAINER_NAME)
            except docker.errors.NotFound:
                try:
                    net = client.networks.get(NET_NAME)
                except docker.errors.NotFound:
                    net = client.networks.create(
                        NET_NAME,
                        driver="bridge",
                        internal=False,
                        options={
                            "com.docker.network.bridge.enable_ip_masquerade": "false",
                        },
                    )

                _CONTAINER = client.containers.run(
                    IMAGE,
                    name=CONTAINER_NAME,
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
                    network=net.name,
                    dns=[],
                    environment={"SERVER_PORT": str(mcp_server_port)},
                )

        self.container = _CONTAINER
        self.mcp_server_port = mcp_server_port
        self.session_id = session_id
        self.interpreter_id: str = interpreter_id or get_unique_id()
        self.tool_names = tool_names
        self.tools_are_checked = False
        self.url = self._get_url()

    async def invoke(self, code: str) -> ExecResult:
        await self._check_tools()
        await self._wait_for_ready()
        result = await self._call_exec(code)
        return result

    async def _check_tools(self) -> None:
        if not self.tool_names or self.tools_are_checked:
            return
        available_tools = await fetch_tools(f"http://localhost:{self.mcp_server_port}")
        available_tool_names = [tool.name for tool in available_tools]
        for tool_name in self.tool_names:
            if tool_name not in available_tool_names:
                raise ValueError(f"Tool {tool_name} not found in MCP server")
        self.tools_are_checked = True

    async def _call_exec(self, code: str) -> ExecResult:
        payload = {
            "code": textwrap.dedent(code),
            "session_id": self.session_id,
            "tool_names": self.tool_names,
            "interpreter_id": self.interpreter_id,
        }

        async with httpx.AsyncClient(limits=httpx.Limits(keepalive_expiry=0)) as client:
            # TODO: cancel internal requests after timeout
            # TODO: proper error for timeout
            resp = await client.post(f"{self.url}/exec", json=payload, timeout=EXEC_TIMEOUT)
            resp.raise_for_status()
            out = resp.json()
            result: ExecResult = ExecResult.model_validate(out)

        if result.stdout:
            result.stdout = truncate_content(result.stdout)
        if result.error:
            result.error = truncate_content(result.error)
        if result.result and isinstance(result.result, str):
            if not is_correct_json(result.result):
                result.result = truncate_content(result.result)
        return result

    def _get_url(self) -> str:
        assert self.container is not None
        self.container.reload()
        ports = self.container.attrs["NetworkSettings"]["Ports"]
        mapping = ports["8000/tcp"][0]
        return f"http://localhost:{mapping['HostPort']}"

    async def cleanup(self) -> None:
        payload = {
            "interpreter_id": self.interpreter_id,
        }
        async with httpx.AsyncClient(limits=httpx.Limits(keepalive_expiry=0)) as client:
            response = await client.post(
                f"{self.url}/cleanup", json=payload, timeout=CLEANUP_TIMEOUT
            )
            response.raise_for_status()

    async def _wait_for_ready(self, max_wait: int = 60) -> None:
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                output = await self._call_exec("print('ready')")
                assert output.stdout.strip() == "ready"
                return
            except (httpx.RequestError, httpx.TimeoutException, AssertionError):
                pass
            await asyncio.sleep(0.1)
        raise RuntimeError("Container failed to become ready within timeout")
