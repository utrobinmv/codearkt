import os

from phoenix.otel import register
from dotenv import load_dotenv

from codearkt.server import run_server
from codearkt.otel import CodeActInstrumentor
from .agents import get_manager

load_dotenv()

MCP_CONFIG = {
    "mcpServers": {"academia": {"url": "http://0.0.0.0:5056/mcp", "transport": "streamable-http"}}
}
PHOENIX_URL = os.getenv("PHOENIX_URL", "http://localhost:6006")
PHOENIX_PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", "codearkt")

register(
    project_name=PHOENIX_PROJECT_NAME,
    endpoint=f"{PHOENIX_URL}/v1/traces",
    auto_instrument=True,
)
CodeActInstrumentor().instrument()


def main() -> None:
    agent = get_manager()
    run_server(agent, MCP_CONFIG)


if __name__ == "__main__":
    main()
