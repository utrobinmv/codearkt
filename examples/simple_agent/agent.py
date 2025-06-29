import os
from dotenv import load_dotenv

from codearkt.codeact import CodeActAgent, Prompts
from codearkt.llm import LLM

load_dotenv()


def get_simple_agent() -> CodeActAgent:
    llm = LLM(model_name="gpt-4o-mini", base_url="https://api.openai.com/v1", api_key=os.getenv("OPENAI_API_KEY"))
    prompts = Prompts.load("codearkt/prompts/codeact.yaml")
    return CodeActAgent(
        name="simple_agent",
        description="A simple agent",
        llm=llm,
        prompts=prompts,
    )
