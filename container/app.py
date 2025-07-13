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

_globals: Dict[str, Any] = {"__name__": "__main__"}
_tools_are_fetched = False


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
    global _tools_are_fetched
    if not _tools_are_fetched and payload.tool_names:
        tools = await fetch_tools(tool_names=payload.tool_names)
        for tool_name, tool_fn in tools.items():
            _globals[tool_name] = tool_fn
            if payload.session_id and tool_name.startswith("agent__"):
                _globals[tool_name] = partial(
                    tool_fn,
                    session_id=payload.session_id,
                )
        _tools_are_fetched = True

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _execute_code, payload.code, _globals)

    return result
