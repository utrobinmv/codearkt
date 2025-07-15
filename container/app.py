import io
import contextlib
import traceback
import asyncio
import ast
from typing import Dict, Any, Optional, List
from functools import partial

from fastapi import FastAPI
from pydantic import BaseModel

from tools import fetch_tools  # type: ignore

app = FastAPI(title="CodeArkt code runtime")

_tools: Dict[str, Any] = {}
_globals: Dict[str, Any] = {"__name__": "__main__"}


class Payload(BaseModel):  # type: ignore
    code: str
    tool_names: List[str]
    session_id: Optional[str] = None


class ExecResult(BaseModel):  # type: ignore
    stdout: str
    error: str
    result: Any | None = None


def _exec_with_return(
    code: str, globals: Dict[str, Any], locals: Dict[str, Any] | None = None
) -> Any:
    a = ast.parse(code)
    last_expression = None
    if a.body:
        if isinstance(a_last := a.body[-1], ast.Expr):
            last_expression = ast.unparse(a.body.pop())
        elif isinstance(a_last, ast.Assign):
            last_expression = ast.unparse(a_last.targets[0])
        elif isinstance(a_last, (ast.AnnAssign, ast.AugAssign)):
            last_expression = ast.unparse(a_last.target)
    exec(ast.unparse(a), globals, locals)
    if last_expression:
        return eval(last_expression, globals, locals)
    return None


def _execute_code(
    code: str,
    globals_dict: Dict[str, Any],
) -> ExecResult:
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            result = _exec_with_return(code, globals_dict)
        return ExecResult(stdout=buf.getvalue(), error="", result=result)
    except Exception:
        return ExecResult(stdout=buf.getvalue(), error=traceback.format_exc(), result=None)


@app.post("/exec")  # type: ignore
async def exec_code(payload: Payload) -> ExecResult:
    # Cache tools
    global _tools
    if not _tools and payload.tool_names:
        _tools = await fetch_tools()

    # Get current tools
    current_tools = {tool_name: _tools[tool_name] for tool_name in payload.tool_names}
    for tool_name in payload.tool_names:
        assert tool_name in current_tools, f"Tool {tool_name} not found"

    for tool_name, tool_fn in current_tools.items():
        _globals[tool_name] = tool_fn
        if payload.session_id and tool_name.startswith("agent__"):
            _globals[tool_name] = partial(
                tool_fn,
                session_id=payload.session_id,
            )

    # Remove unused tools
    unused_tools = set(_tools.keys()) - set(current_tools.keys())
    for tool_name in unused_tools:
        if tool_name in _globals:
            del _globals[tool_name]

    # Execute code
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _execute_code, payload.code, _globals)

    return result
