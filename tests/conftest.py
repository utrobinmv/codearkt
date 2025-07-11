import asyncio
import threading
import time
import logging
from typing import Generator
from contextlib import suppress

import pytest
import uvicorn
from dotenv import load_dotenv
from academia_mcp.server import http_app as academia_app

from codearkt.llm import LLM

load_dotenv()
for name in ("httpx", "mcp", "openai"):
    logging.getLogger(name).setLevel(logging.WARNING)


@pytest.fixture
def gpt_4o_mini() -> LLM:
    return LLM(model_name="gpt-4o-mini")


class MCPServerTest:
    def __init__(self, port: int = 5055) -> None:
        self.port = port
        self.app = academia_app
        self.server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._started = threading.Event()
        for name in ("uvicorn", "uvicorn.error"):
            logging.getLogger(name).setLevel(logging.ERROR)
        logging.getLogger("uvicorn.access").setLevel(logging.CRITICAL)

    def start(self) -> None:
        config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="error",
            access_log=False,
            lifespan="on",
        )
        self.server = uvicorn.Server(config)

        def _run() -> None:
            async def _serve() -> None:
                assert self.server is not None
                await self.server.serve()

            with suppress(asyncio.CancelledError):
                asyncio.run(_serve())

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

        deadline = time.time() + 30
        while time.time() < deadline:
            if self.server.started:
                self._started.set()
                break
            time.sleep(0.05)
        if not self._started.is_set():
            raise RuntimeError("Mock MCP server failed to start within 30 s")

    def stop(self) -> None:
        if self.server:
            self.server.should_exit = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def is_running(self) -> bool:
        return self._started.is_set() and self._thread is not None and self._thread.is_alive()


@pytest.fixture(scope="session")
def mcp_server_test() -> Generator[MCPServerTest, None, None]:
    server = MCPServerTest(port=5055)
    server.start()
    yield server
    server.stop()
