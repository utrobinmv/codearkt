from codearkt.server import run_server
from .agent import get_multi_agent


MCP_CONFIG = {
    "mcpServers": {
        "academia": {
            "url": "http://0.0.0.0:5056/mcp",
            "transport": "streamable-http"
        }
    }
}


def main() -> None:
    agent = get_multi_agent()
    run_server(agent, MCP_CONFIG)


if __name__ == "__main__":
    main()