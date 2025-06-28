from codearkt.session import Session


class PythonExecutor:
    def __init__(self) -> None:
        self.session = Session()

    def invoke(self, code_action: str) -> str:
        output = self.session.run(code_action)
        return output
