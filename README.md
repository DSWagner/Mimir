# Mimir — Blockscout Agent for Agentverse

> Autonomous blockchain analyst that uses **Blockscout MCP** tools and **ASI's asi1-extended LLM** to investigate and explain on-chain activity across multiple EVM chains.

---

## ✨ What is Mimir?

**Mimir** is an advanced autonomous analysis agent built with **uAgents** that connects to the **Blockscout MCP** (Modular Chain Protocol) server to retrieve verified, explorer‑grade data (transactions, token transfers, contract metadata/ABI, etc.).  
It reasons step‑by‑step with **ASI’s `asi1-extended`** model and produces clear, human‑readable explanations of what’s happening on‑chain.

Mimir is designed first for **ASI Agentverse**, but it can run locally for development and can be messaged via the standard **uAgents Chat Protocol**.

---

## 🧠 Key Features (mapped to code)

- **Async‑safe LLM + MCP orchestration** (LLM calls wrapped with `asyncio.to_thread`)
- **Unlock‑first rule**: always calls `__unlock_blockchain_analysis__` as the very first tool
- **Deep system prompt** with:
  - explicit final‑answer sentinel `<<<FINAL_ANSWER>>>`
  - pagination & time-window tactics
  - binary search strategy for historical ranges
  - tool argument validation rules (e.g., `chain_id` must be a **string**)
- **Auto‑refresh MCP tools every 12h** and caches tools in storage
- **Conversation trimming** (`MAX_HISTORY=10`) to keep context focused
- **Tool schema validation** + **autofill `chain_id`** (string) using text hints & defaults
- **Safe tool execution** with retries/backoff; errors are returned as `TOOL_RESULT`
- **Sanitized logging** (masks keys/tokens)
- **Instruction injection** from unlock tool results
- **SSE parsing** for MCP responses with idle‑timeout safeguards

> See the header docstring of `agent.py` for a concise checklist of the above.

---

## 🧩 Architecture Overview

```
User (Agentverse chat or any uAgents client)
        │
        ▼
Mimir Agent (uAgents)
  ├─ LLM: ASI chat completions (model: asi1-extended)
  ├─ System prompt builder (+ unlock instruction injection)
  ├─ Tool runner (validation, retries, backoff, pagination)
  └─ Context manager (history trim, final sentinel)
        │
        ▼
Blockscout MCP Server (JSON‑RPC over HTTP/S + SSE)
  └─ Tools: address/tx/token/ABI/ENS/etc.
```

- **ASI endpoint**: `https://api.asi1.ai/v1/chat/completions`
- **MCP endpoint (default)**: `https://mcp.blockscout.com/mcp` (JSON‑RPC; not REST)
- **Chat protocol**: [`uagents_core.contrib.protocols.chat`](https://github.com/fetchai/uAgents) (manifest is published by the agent)

---

## 📦 Requirements

- **Python 3.10+**
- `uagents`, `requests`, `python-dotenv`

> If you use a private MCP instance, it must support the **MCP JSON‑RPC** interface (the code calls `tools/list` and `tools/call`), and ideally **SSE** for streamed results.

---

## ⚙️ Environment Variables

Create a `.env` in the project root if you prefer environment‑file configuration.

| Variable             | Description                                  | Default                          |
| -------------------- | -------------------------------------------- | -------------------------------- |
| `ASI_API_KEY`        | Primary ASI key for `asi1-extended`          | —                                |
| `ASI_ONE_API_KEY`    | Fallback env var name for the ASI key        | —                                |
| `BLOCKSCOUT_MCP_URL` | MCP JSON‑RPC base URL                        | `https://mcp.blockscout.com/mcp` |
| `DEFAULT_CHAIN_ID`   | Chain used when not inferable from user text | `1` (Ethereum)                   |

> The runtime loader checks: `ASI_API_KEY` → `ASI_ONE_API_KEY` → `ctx.storage["asi_api_key"]`.

---

### Messaging the agent locally

Mimir publishes the **uAgents chat protocol** manifest. You can message it from another uAgents script (or from Agentverse). Minimal example:

```python
# examples/ping.py
from uagents import Agent, Context, Protocol
from uagents_core.contrib.protocols.chat import ChatMessage, TextContent, chat_protocol_spec

SENDER = Agent(name="tester", seed="tester_seed", mailbox=True)
chat_protocol = Protocol(spec=chat_protocol_spec)
SENDER.include(chat_protocol)

@chat_protocol.on_interval(period=2)
async def send(_ctx: Context):
    await SENDER.send("<MIMIR_AGENT_ADDRESS>", ChatMessage(content=[TextContent(type="text", text="Analyze transfers for 0xd8dA6... on Ethereum")]))

if __name__ == "__main__":
    SENDER.run()
```

Replace `<MIMIR_AGENT_ADDRESS>` with the address from Mimir’s startup logs.

---

## 🔐 Behavior & Guardrails

- **Unlock‑first**: refuses to proceed if the first step/tool isn’t `__unlock_blockchain_analysis__`.
- **Tool parameter rules**: validates required fields; autofills **`chain_id` as a string** using chain hints (`ethereum → "1"`, `polygon → "137"`, `arbitrum → "42161"`, `base → "8453"`, `bsc → "56"`, etc.).
- **Pagination**: if a tool result returns a `pagination.next_call`, Mimir continues until done or a reasonable limit.
- **Time‑based search**: supports `age_from` / `age_to` in ISO‑8601; can binary‑search long histories.
- **Security**: never exposes system prompts; masks secrets in logs; returns tool errors inside `TOOL_RESULT` instead of crashing.
- **Context limits**: trims conversation to `MAX_HISTORY=10`; final answers end with the sentinel `<<<FINAL_ANSWER>>>` (removed before sending).

---

## 🧪 Troubleshooting

- **“ASI_API_KEY not configured”** — Ensure the key is present in environment or `.env`.
- **MCP connection failed** — Verify `BLOCKSCOUT_MCP_URL`, network access, and that your MCP server supports **JSON‑RPC** (`tools/list`, `tools/call`) and responds with JSON or SSE.
- **Finalizing without unlock** — Check that the MCP has the `__unlock_blockchain_analysis__` tool; the agent attempts to call it first.
- **Chain ID type errors** — All tool `chain_id` parameters must be **strings** (e.g., `"1"`). The agent will warn if it detects numeric usage.

---

## 🗺️ Configuration knobs (edit in `agent.py`)

- `ASI_ENDPOINT = "https://api.asi1.ai/v1/chat/completions"`
- `ASI_MODEL = "asi1-extended"`
- `MCP_URL` from `BLOCKSCOUT_MCP_URL` (default `"https://mcp.blockscout.com/mcp"`)
- `MCP_TIMEOUT = 90`
- `MAX_TOOL_STEPS = 25`
- `MAX_HISTORY = 10`
- `FINAL_SENTINEL = "<<<FINAL_ANSWER>>>"`

---

## 📄 License

MIT (replace with your actual license if different).
