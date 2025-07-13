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
answer1 = json.loads(arxiv_download(paper_id="2506.15003"))["title"]
print("Answer 1:", answer1, end="")
"""

PIP_INSTALL_SNIPPET = """
import subprocess
proc = subprocess.run(['pip', 'install', 'transformers'], capture_output=True)
print(proc.stdout)
print(proc.stderr)
"""


@pytest.mark.asyncio(loop_scope="function")
async def test_python_executor_basic(
    default_python_executor: PythonExecutor,
) -> None:
    result1 = await default_python_executor.invoke(SNIPPET_1)
    result2 = await default_python_executor.invoke(SNIPPET_2)
    assert result1.stdout == "Answer 1"
    assert result2.stdout == "Variable still here: Answer 1"


@pytest.mark.asyncio(loop_scope="function")
async def test_python_executor_mcp_invokation_no_tools(
    mcp_server_test: MCPServerTest,
    default_python_executor: PythonExecutor,
) -> None:
    _ = mcp_server_test
    result = await default_python_executor.invoke(SNIPPET_3)
    assert (
        result.stdout
        != "Answer 1: Effect of surface magnetism on the x-ray spectra of hollow atoms"
    )
    assert result.error
    assert "Error" in result.error


@pytest.mark.asyncio(loop_scope="function")
async def test_python_executor_pip_install(
    default_python_executor: PythonExecutor,
) -> None:
    result = await default_python_executor.invoke(PIP_INSTALL_SNIPPET)
    assert "ERROR" in result.stdout, result.stdout


@pytest.mark.asyncio(loop_scope="function")
async def test_python_executor_result_expr(
    default_python_executor: PythonExecutor,
) -> None:
    result = await default_python_executor.invoke("r = 4\nr")
    assert result.result == 4


@pytest.mark.asyncio(loop_scope="function")
async def test_python_executor_result_dict(
    default_python_executor: PythonExecutor,
) -> None:
    result = await default_python_executor.invoke("r = {'a': 4}\nr")
    assert result.result == {"a": 4}


@pytest.mark.asyncio(loop_scope="function")
async def test_python_executor_mcp_invokation(
    mcp_server_test: MCPServerTest,
) -> None:
    _ = mcp_server_test
    executor = PythonExecutor(tool_names=["arxiv_download"])
    result = await executor.invoke(SNIPPET_3)
    assert (
        result.stdout
        == "Answer 1: Effect of surface magnetism on the x-ray spectra of hollow atoms"
    )


@pytest.mark.asyncio(loop_scope="function")
async def test_python_executor_non_existing_tool() -> None:
    executor = PythonExecutor(tool_names=["arxiv_download"])
    with pytest.raises(ValueError):
        await executor.invoke(SNIPPET_3)
