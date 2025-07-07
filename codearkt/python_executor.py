from typing import Tuple

from codearkt.session import Session


class PythonExecutor:
    def __init__(self, session_id: str) -> None:
        self.session = Session(session_id)

    async def invoke(self, code_action: str) -> Tuple[str, str]:
        output, logs = await self.session.run(code_action)
        return output, logs
