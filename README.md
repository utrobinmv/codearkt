# CodeArkt ⚡️🛠️

[![PyPI](https://img.shields.io/pypi/v/codearkt?label=PyPI%20package)](https://pypi.org/project/codearkt/)
[![CI](https://github.com/IlyaGusev/codearkt/actions/workflows/python.yml/badge.svg)](https://github.com/IlyaGusev/codearkt/actions/workflows/python.yml)
[![License](https://img.shields.io/github/license/IlyaGusev/codearkt)](LICENSE)
[![Stars](https://img.shields.io/github/stars/IlyaGusev/codearkt?style=social)](https://github.com/IlyaGusev/codearkt/stargazers)

> **CodeArkt** is a battery-included implementation of the **CodeAct** multi-agent architecture. Ship autonomous coding agents that can reason, write, execute & iterate over code – all from a single Python package.

---

## ✨ Why CodeArkt?

* **True multi-agent orchestration** – coordinate unlimited specialist agents with a single *manager* agent.
* **Live Python sandbox** – secure, ephemeral execution environment with real-time streaming of stdout/stderr back to the LLM.
* **Event-driven core** – subscribe to granular `AgentEvent`s (start, output, tool call, error) and build your own dashboards.
* **First-class tool ecosystem** – auto-discover & register [MCP](https://github.com/academia-org/mcp) tools or your own custom functions.
* **Drop-dead simple UI** – launch an elegant Gradio chat in one line: `python -m codearkt.gradio`.
* **Production ready** – typed codebase (`mypy --strict`), CI, tests, Docker & Apache-2.0 license.

---

## 🚀 Quick Start

Install the package:
```bash
pip install codearkt  # requires Python ≥ 3.12
```

Write a simple agent:
```python
from codearkt.codeact import CodeActAgent
from codearkt.llm import LLM
from codearkt.server import run_server

# Use your own or remote MCP servers
mcp_config = {
    "mcpServers": {"academia": {"url": "http://0.0.0.0:5056/mcp", "transport": "streamable-http"}}
}

# Create an agent definition
agent = CodeActAgent(
    name="manager",
    description="A simple agent",
    llm=LLM(model_name="deepseek/deepseek-chat-v3-0324"),
    tool_names=["arxiv_download", "arxiv_search"],
)

# Run the server with MCP proxy and agentic endpoints
run_server(agent, mcp_config)
```

Within seconds you will see agents collaborating, executing Python snippets, and streaming the results back to your console / web chat.

---

## 🧩 Feature Overview

| Area | Highlights |
|------|------------|
| Agents | Hierarchical manager / worker pattern, pluggable prompts, configurable iteration limits |
| Tools | Automatic discovery via MCP registry, Python execution (`python_interpreter`) |
| Execution | Sandboxed temp directory, timeout, streamed chunks, cleanup hooks |
| Observability | `AgentEventBus` publishes JSON events – integrate with logs, websockets or GUI |
| UI | Responsive Gradio Blocks chat with stop button, syntax-highlighted code & output panels |
| Extensibility | Compose multiple `CodeActAgent` instances, add your own LLM backend, override prompts |

---

## 📖 Documentation

For now, explore the well-typed source code.

---

## 🛠️ Project Structure

```
codearkt/
├─ codeact.py          # Core agent logic
├─ python_executor.py  # Secure sandbox for arbitrary code
├─ event_bus.py        # Pub/Sub for agent events
├─ gradio.py           # Optional web UI
└─ ...
examples/
└─ multi_agent/        # End-to-end usage demos
```

---

## 🗺️ Roadmap

- [ ] Otel integration
- [ ] CI for the Docker image
- [ ] Full documentation & tutorials
- [ ] Test coverage

---

## 🤝 Contributing

Pull requests are welcome! Please:

1. Fork the repo & create your branch: `git checkout -b feature/my-feature`  
2. Install dev deps: `make install`
3. Run the linter & tests: `make validate && make test`  
4. Submit a PR and pass the CI.  

Join the discussion in **[Discussions](https://github.com/IlyaGusev/codearkt/discussions)** or open an **[Issue](https://github.com/IlyaGusev/codearkt/issues)**.

---

## 📝 License

`CodeArkt` is released under the Apache License 2.0 – see the [LICENSE](LICENSE) file for details.