from codearkt.mcp_server import run_mcp_server
from .agent import get_simple_agent


MCP_CONFIG = {
    "mcpServers": {
        "academia": {
            "url": "http://0.0.0.0:5056/mcp",
            "transport": "streamable-http"
        }
    }
}


def main() -> None:
    agent = get_simple_agent()
    run_mcp_server([agent], MCP_CONFIG)


if __name__ == "__main__":
    main()