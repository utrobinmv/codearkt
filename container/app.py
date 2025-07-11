import io
import contextlib
import traceback
import asyncio
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


def _execute_code(
    code: str,
    globals_dict: Dict[str, Any],
) -> ExecResult:
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, globals_dict)
        return ExecResult(stdout=buf.getvalue(), error="")
    except Exception:
        return ExecResult(stdout=buf.getvalue(), error=traceback.format_exc())


@app.post("/exec")  # type: ignore
async def exec_code(payload: Payload) -> ExecResult:
    global _tools_are_fetched
    if not _tools_are_fetched:
        try:
            tools = await fetch_tools(tool_names=payload.tool_names)
            for tool_name, tool_fn in tools.items():
                _globals[tool_name] = tool_fn
                if payload.session_id and tool_name.startswith("agent__"):
                    _globals[tool_name] = partial(
                        tool_fn,
                        session_id=payload.session_id,
                    )
            _tools_are_fetched = True
        except Exception:
            print("Failed to fetch tools")
            print(traceback.format_exc())
            pass

    loop = asyncio.get_event_loop()

    result = await loop.run_in_executor(None, _execute_code, payload.code, _globals)

    return result
