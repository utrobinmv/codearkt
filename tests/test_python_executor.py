import pytest

from codearkt.python_executor import PythonExecutor
from tests.conftest import MCPServerTest

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


@pytest.mark.asyncio(loop_scope="session")
async def test_python_executor_basic() -> None:
    executor = PythonExecutor("testid", tool_names=[])
    result1 = await executor.invoke(SNIPPET_1)
    result2 = await executor.invoke(SNIPPET_2)
    assert result1 == "Answer 1"
    assert result2 == "Variable still here: Answer 1"


@pytest.mark.asyncio(loop_scope="session")
async def test_python_executor_mcp_invokation(mcp_server_test: MCPServerTest) -> None:
    executor = PythonExecutor("testid", tool_names=["arxiv_download"])
    result = await executor.invoke(SNIPPET_3)
    assert result == "Answer 1: Effect of surface magnetism on the x-ray spectra of hollow atoms"


@pytest.mark.asyncio(loop_scope="session")
async def test_python_executor_mcp_invokation_no_tools(mcp_server_test: MCPServerTest) -> None:
    executor = PythonExecutor("testid", tool_names=[])
    result = await executor.invoke(SNIPPET_3)
    assert result != "Answer 1: Effect of surface magnetism on the x-ray spectra of hollow atoms"
    assert "Error" in result
