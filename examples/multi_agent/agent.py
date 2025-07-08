import os
from pathlib import Path
from dotenv import load_dotenv

from codearkt.codeact import CodeActAgent, Prompts
from codearkt.llm import LLM

load_dotenv()

current_dir = Path(__file__).parent


LIBRARIAN_DESCRIPTION = """This team member runs gets and analyzes information from papers.
He has access to ArXiv, Semantic Scholar, ACL Anthology, and web search.
Ask him any questions about papers and web articles.
Give him your task as an argument. Follow the task format described above, include all the details."""


def get_librarian() -> CodeActAgent:
    llm = LLM(model_name="gpt-4o-mini", base_url="https://api.openai.com/v1", api_key=os.getenv("OPENAI_API_KEY"))
    prompts = Prompts.load(current_dir / "prompts" / "librarian.yaml")
    return CodeActAgent(
        name="librarian",
        description=LIBRARIAN_DESCRIPTION,
        llm=llm,
        prompts=prompts,
    )


def get_multi_agent() -> CodeActAgent:
    llm = LLM(model_name="gpt-4o-mini", base_url="https://api.openai.com/v1", api_key=os.getenv("OPENAI_API_KEY"))
    prompts = Prompts.load("codearkt/prompts/codeact.yaml")
    return CodeActAgent(
        name="manager",
        description="A manager agent",
        llm=llm,
        prompts=prompts,
        managed_agents=[get_librarian()],
    )