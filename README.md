# AI Council (Local Multi-Agent System)

AI Council is a local, privacy-first multi-agent system powered by **Ollama** and open-source LLMs.
Each agent has a unique personality and answers the user’s question.
A separate judge model evaluates all answers and selects the best response.

Everything runs **100% offline** and all data stays on your machine.

---

## Features

- Multiple AI agents with different personalities
- Independent "judge" model selects the best answer
- Optional performance-based elimination and replacement system
- Runs entirely on local hardware (GPU-accelerated)
- Easy to extend with more agents or models
- No cloud services, no telemetry, no uploads

---

## Requirements

- Python 3.10+
- Ollama installed: https://ollama.com/download
- At least one model pulled, e.g.:

```bash
ollama pull llama3.1:8b
