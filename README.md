# OrchestrAI

**The open-source MCP orchestration layer for multi-model software engineering.**

OrchestrAI exposes a single [Model Context Protocol](https://modelcontextprotocol.io) server that routes software engineering tasks across the best available AI models — Anthropic, OpenAI, Gemini, and local models via Ollama or vLLM — automatically assigning each role (Planner, Coder, Tester, Reviewer, Judge) to the model most capable of handling it.

---

## Why OrchestrAI?

| Problem | OrchestrAI Solution |
|---------|-------------------|
| One model doing everything | Role-based routing to specialist models |
| No verification of AI output | Built-in lint, test, and type-check runners |
| Vendor lock-in | Provider-agnostic adapter layer |
| Privacy concerns | `secret` tier forces local-only routing |
| Black-box decisions | Full artifact + trace system with provenance |

---

## Architecture

```
MCP Client (Claude, Cursor, etc.)
        │
        ▼
 ┌─────────────────┐
 │   MCP Server    │  tools, resources, prompts
 └────────┬────────┘
          │
 ┌────────▼────────┐
 │   Orchestrator  │  intake → route → execute → trace
 └────────┬────────┘
          │
    ┌─────┴──────┐
    │  Mode Map  │
    │ ┌────────┐ │
    │ │ PCR    │ │  planner_coder_reviewer
    │ │ Draft  │ │  parallel_draft
    │ │ Impl   │ │  impl_tester
    └─┴────────┴─┘
          │
 ┌────────▼────────┐
 │ Routing Engine  │  capability registry + policy
 └────────┬────────┘
          │
  ┌───────┴────────┐
  │   Providers    │
  │ Anthropic      │  claude-opus-4-6, sonnet-4-6, haiku-4-5
  │ OpenAI         │  gpt-4.1, gpt-4o, o4-mini
  │ Gemini         │  gemini-2.5-pro, gemini-2.0-flash
  │ Local (Ollama) │  codellama, qwen2.5-coder, etc.
  └────────────────┘
```

---

## Quick Start

### Install

```bash
pip install orchestrai
# or from source:
git clone https://github.com/musaceylan/OrchestrAI
cd OrchestrAI && pip install -e .
```

### Configure Claude Desktop / Cursor

Add to your MCP client config:

```json
{
  "mcpServers": {
    "orchestrai": {
      "command": "orchestrai",
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "OPENAI_API_KEY": "sk-...",
        "GEMINI_API_KEY": "AI..."
      }
    }
  }
}
```

### Run directly

```bash
ANTHROPIC_API_KEY=... orchestrai --log-format console
```

---

## Orchestration Modes

| Mode | Best For | Flow |
|------|----------|------|
| `planner_coder_reviewer` | Complex features, bugfixes | Plan → Code+Test → Lint → Review |
| `parallel_draft` | Comparing implementations | N×Code in parallel → Judge → Best |
| `impl_tester` | TDD, test generation, docs | Code+Test in parallel → Review |

---

## MCP Tools

| Tool | Description |
|------|-------------|
| `submit_task` | Submit a task for orchestrated execution |
| `get_task_result` | Get the winning patch + review verdict |
| `inspect_registry` | List all providers and model capabilities |
| `inspect_agents` | See which models are assigned to each role |
| `inspect_artifacts` | Browse patches, tests, reviews for a task |
| `inspect_trace` | Full execution timeline with timing |
| `compare_candidates` | Run judge on parallel_draft candidates |
| `rerun_with_policy` | Re-run with different privacy/cost constraints |
| `list_available_models` | All discovered models with strengths |
| `probe_providers` | Re-probe providers and refresh registry |

## Built-in Prompts

`bugfix` · `feature` · `refactor` · `review` · `test_generation` · `docs` · `local_only`

---

## Privacy Tiers

```
secret       → local models only (Ollama, vLLM) — nothing leaves your machine
confidential → local + enterprise APIs only
internal     → any provider without external telemetry (default)
public       → all providers allowed
```

Set via `user_preferences: {privacy_level: "secret"}` in `submit_task`.

`local_only` means machine-loopback only: `localhost`, names beneath `.localhost`,
IPv4 `127.0.0.0/8`, and IPv6 `::1`. LAN/private-network and remote OpenAI-compatible
endpoints are classified as `public` and remain available only when the selected
privacy policy permits public routing.

---

## Local Models (Privacy-First)

Start Ollama and pull a model:

```bash
ollama serve
ollama pull qwen2.5-coder:7b
```

OrchestrAI auto-discovers local models at `http://localhost:11434`. Add custom endpoints in config:

```yaml
local_providers:
  - name: vllm-gpu
    base_url: http://localhost:8000
```

---

## Development

```bash
# Install with dev extras
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check src/
```

---

## License

MIT
