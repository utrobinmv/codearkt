import os
from pathlib import Path

from phoenix.otel import register
from dotenv import load_dotenv

from codearkt.codeact import CodeActAgent, Prompts
from codearkt.llm import LLM
from codearkt.server import run_server
from codearkt.otel import CodeActInstrumentor

load_dotenv()

current_dir = Path(__file__).parent

PHOENIX_URL = os.getenv("PHOENIX_URL", "http://localhost:6006")
PHOENIX_PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", "codearkt")
EXA_API_KEY = os.getenv("EXA_API_KEY", "")
MCP_CONFIG = {
    "mcpServers": {
        "academia": {"url": "http://0.0.0.0:5056/mcp", "transport": "streamable-http"},
        "exa": {
            "url": f"https://mcp.exa.ai/mcp?exaApiKey={EXA_API_KEY}",
            "transport": "streamable-http",
        },
    }
}

LIBRARIAN_DESCRIPTION = """This team member runs gets and analyzes information from papers.
He has access to ArXiv, Semantic Scholar, ACL Anthology, and web search.
Ask him any questions about papers and web articles.
Give him your task as an only string argument. Follow the task format described above, include all the details."""


def get_librarian() -> CodeActAgent:
    llm = LLM(model_name="deepseek/deepseek-chat-v3-0324")
    prompts = Prompts.load(current_dir / "librarian.yaml")
    return CodeActAgent(
        name="librarian",
        description=LIBRARIAN_DESCRIPTION,
        llm=llm,
        prompts=prompts,
        tool_names=[
            "academia_arxiv_download",
            "academia_arxiv_search",
            "exa_web_search_exa",
            "exa_crawling_exa",
        ],
        planning_interval=5,
    )


def get_manager() -> CodeActAgent:
    llm = LLM(model_name="deepseek/deepseek-chat-v3-0324")
    return CodeActAgent(
        name="manager",
        description="A manager agent",
        llm=llm,
        managed_agents=[get_librarian()],
        tool_names=[],
        planning_interval=5,
    )


def main() -> None:
    register(
        project_name=PHOENIX_PROJECT_NAME,
        endpoint=f"{PHOENIX_URL}/v1/traces",
        auto_instrument=True,
    )
    CodeActInstrumentor().instrument()
    agent = get_manager()
    run_server(agent, MCP_CONFIG)


if __name__ == "__main__":
    main()
