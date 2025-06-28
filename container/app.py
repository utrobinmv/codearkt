import io
import contextlib
import traceback
import asyncio
from typing import Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel

from tools import fetch_tools  # type: ignore

app = FastAPI(title="CodeArkt code runtime")

_globals: Dict[str, Any] = {"__name__": "__main__"}
_tools_are_fetched = False


class Code(BaseModel):  # type: ignore
    code: str


class ExecResult(BaseModel):  # type: ignore
    stdout: str
    error: str


def _execute_code(code: str, globals_dict: Dict[str, Any]) -> ExecResult:
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, globals_dict)
        return ExecResult(stdout=buf.getvalue(), error="")
    except Exception:
        return ExecResult(stdout=buf.getvalue(), error=traceback.format_exc())


@app.post("/exec")  # type: ignore
async def exec_code(payload: Code) -> ExecResult:
    global _tools_are_fetched
    if not _tools_are_fetched:
        tools = await fetch_tools()
        for tool_name, tool_fn in tools.items():
            _globals[tool_name] = tool_fn
        _tools_are_fetched = True

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _execute_code, payload.code, _globals)
    return result
