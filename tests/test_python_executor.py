import pytest

from codearkt.python_executor import PythonExecutor


SNIPPET_1 = """
answer = "Answer 1"
print(answer, end="")
"""

SNIPPET_2 = """
print("Variable still here:", answer, end="")
"""

SNIPPET_3 = """
import json
answer = json.loads(arxiv_download(paper_id="2506.15003"))["title"]
print("Answer 1:", answer, end="")
"""


@pytest.mark.asyncio(scope="session")
async def test_python_executor_basic():
    executor = PythonExecutor("testid")
    result1 = await executor.invoke(SNIPPET_1)
    result2 = await executor.invoke(SNIPPET_2)
    assert result1 == "Answer 1"
    assert result2 == "Variable still here: Answer 1"


@pytest.mark.asyncio(scope="session")
async def test_python_executor_mcp_invokation():
    executor = PythonExecutor("testid")
    result = await executor.invoke(SNIPPET_3)
    assert result == "Answer 1: A Unified Framework for Multi-Agent Reinforcement Learning"