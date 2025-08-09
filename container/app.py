import io
import contextlib
import traceback
import asyncio
import ast
from typing import Dict, Any, Optional, List, Callable
from functools import partial
import multiprocessing
import time
import logging

from fastapi import FastAPI, Request
from pydantic import BaseModel

from tools import fetch_tools  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(title="CodeArkt code runtime")
WORKERS: Dict[str, "Worker"] = {}


@app.middleware("http")  # type: ignore
async def log_requests(request: Request, call_next: Callable[[Request], Any]) -> Any:
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000
    logging.info(
        "%s %s completed in %.2f ms with status %d",
        request.method,
        request.url.path,
        duration_ms,
        response.status_code,
    )
    return response


class Payload(BaseModel):  # type: ignore
    code: str
    tool_names: List[str]
    tool_server_port: Optional[int] = None
    interpreter_id: Optional[str] = None
    session_id: Optional[str] = None


class CleanupPayload(BaseModel):  # type: ignore
    interpreter_id: str


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


def _worker_main(request_q: multiprocessing.Queue, response_q: multiprocessing.Queue) -> None:  # type: ignore
    import asyncio

    _tools: Dict[str, Any] = {}
    _globals: Dict[str, Any] = {"__name__": "__main__"}

    while True:
        item = request_q.get()
        if item is None:
            break
        code: str = item["code"]
        tool_server_port: int = item["tool_server_port"]
        tool_names: List[str] = item["tool_names"]
        session_id: Optional[str] = item["session_id"]
        current_tools: Dict[str, Any] = dict()

        if tool_names:
            try:
                if not _tools or any(name not in _tools for name in tool_names):
                    _tools = asyncio.run(fetch_tools(tool_server_port))

                current_tools = {name: _tools[name] for name in tool_names}
                for name, fn in current_tools.items():
                    if session_id and name.startswith("agent__"):
                        _globals[name] = partial(fn, session_id=session_id)
                    else:
                        _globals[name] = fn
            except Exception:
                print("Error fetching tools", traceback.format_exc())
                exec_result = ExecResult(stdout="", error=traceback.format_exc(), result=None)
                response_q.put(exec_result.model_dump())
                continue

        # Remove tools that are no longer requested
        unused = set(_globals.keys()) & set(_tools.keys()) - set(current_tools.keys())
        for name in unused:
            _globals.pop(name, None)

        exec_result = _execute_code(code, _globals)
        response_q.put(exec_result.model_dump())


class Worker:
    def __init__(self) -> None:
        self._request_q: multiprocessing.Queue[Optional[Dict[str, Any]]] = multiprocessing.Queue()
        self._response_q: multiprocessing.Queue[Any] = multiprocessing.Queue()
        self._process = multiprocessing.Process(
            target=_worker_main,
            args=(self._request_q, self._response_q),
            daemon=True,
        )
        self._process.start()

    def exec(
        self,
        code: str,
        tool_server_port: Optional[int],
        tool_names: List[str],
        session_id: Optional[str],
    ) -> Any:
        self._request_q.put(
            {
                "code": code,
                "tool_server_port": tool_server_port,
                "tool_names": tool_names,
                "session_id": session_id,
            }
        )
        return self._response_q.get()

    def terminate(self) -> None:
        self._request_q.put(None)
        if self._process.is_alive():
            try:
                self._process.terminate()
            except Exception:
                pass
            self._process.join(timeout=2)

        if self._process.is_alive():
            try:
                self._process.kill()
            except Exception:
                pass
            self._process.join(timeout=1)

        try:
            self._request_q.close()
        except Exception:
            pass
        try:
            self._response_q.close()
        except Exception:
            pass


def _get_worker(interpreter_id: str) -> Worker:
    if interpreter_id not in WORKERS:
        WORKERS[interpreter_id] = Worker()
    return WORKERS[interpreter_id]


@app.post("/exec")  # type: ignore
async def exec_code(payload: Payload) -> ExecResult:
    interpreter_id = payload.interpreter_id or "default"
    worker = _get_worker(interpreter_id)

    loop = asyncio.get_event_loop()
    result_dict: Dict[str, Any] = await loop.run_in_executor(
        None,
        worker.exec,
        payload.code,
        payload.tool_server_port,
        payload.tool_names,
        payload.session_id,
    )
    result: ExecResult = ExecResult.model_validate(result_dict)
    return result


@app.post("/cleanup")  # type: ignore
async def cleanup(payload: CleanupPayload) -> Dict[str, str]:
    interpreter_id = payload.interpreter_id
    worker = _get_worker(interpreter_id)
    worker.terminate()
    WORKERS.pop(interpreter_id, None)
    return {"status": "ok"}
