from dotenv import load_dotenv

from codearkt.codeact import CodeActAgent
from codearkt.llm import LLM

load_dotenv()


def get_simple_agent() -> CodeActAgent:
    return CodeActAgent(
        name="manager",
        description="A simple agent",
        llm=LLM(model_name="gpt-4o-mini"),
        tool_names=["arxiv_download", "arxiv_search"],
    )
