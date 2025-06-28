from typing import Tuple

from codearkt.session import Session


class PythonExecutor:
    def __init__(self) -> None:
        self.session = Session()

    def invoke(self, code_action: str) -> Tuple[str, str]:
        output, logs = self.session.run(code_action)
        return output, logs
