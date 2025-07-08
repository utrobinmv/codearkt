import os

import pytest
from dotenv import load_dotenv

from codearkt.llm import LLM


load_dotenv()


@pytest.fixture
def gpt_4o_mini() -> LLM:
    return LLM(
        model_name="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key=os.getenv("OPENAI_API_KEY", ""),
    )
