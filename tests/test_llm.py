from codearkt.llm import LLM, ChatMessage


class TestLLM:
    def test_llm_params_override(self) -> None:
        llm = LLM(
            model_name="test-model", temperature=0.25, top_p=0.3, max_tokens=1024, stop=["STOP"]
        )

        assert llm.model_name == "test-model"
        assert llm.params["temperature"] == 0.25
        assert llm.params["top_p"] == 0.3
        assert llm.params["max_tokens"] == 1024
        assert llm.params["stop"] == ["STOP"]

    async def test_llm_astream(self, deepseek: LLM) -> None:
        output = ""
        async for chunk in deepseek.astream(
            [ChatMessage(role="user", content="Output only 'Hello, world!'")],
        ):
            if chunk.content:
                output += chunk.content
        assert "Hello, world!" in output
