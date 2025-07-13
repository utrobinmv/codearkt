from pathlib import Path
from dotenv import load_dotenv

from codearkt.codeact import CodeActAgent, Prompts
from codearkt.llm import LLM

load_dotenv()

current_dir = Path(__file__).parent


LIBRARIAN_DESCRIPTION = """This team member runs gets and analyzes information from papers.
He has access to ArXiv, Semantic Scholar, ACL Anthology, and web search.
Ask him any questions about papers and web articles.
Give him your task as an only string argument. Follow the task format described above, include all the details."""


def get_librarian() -> CodeActAgent:
    llm = LLM(model_name="gpt-4o-mini")
    prompts = Prompts.load(current_dir / "prompts" / "librarian.yaml")
    return CodeActAgent(
        name="librarian",
        description=LIBRARIAN_DESCRIPTION,
        llm=llm,
        prompts=prompts,
        tool_names=["arxiv_download", "arxiv_search"],
    )


def get_manager() -> CodeActAgent:
    llm = LLM(model_name="gpt-4o-mini")
    return CodeActAgent(
        name="manager",
        description="A manager agent",
        llm=llm,
        managed_agents=[get_librarian()],
        tool_names=[],
    )
