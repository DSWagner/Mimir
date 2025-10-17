# Mimir MCP Agent — Local, MCP‑First CLI

This project is a **local command‑line agent** that uses a **Blockscout MCP server** (JSON‑RPC over HTTP) to fetch on‑chain data and an **ASI:One model** to reason about which tools to call and how to synthesize a final answer.

It **does not** expose an HTTP API, and it **does not** use uAgents. You run a Dockerized **Blockscout MCP server**, then run the **Python CLI** and pass your question directly:

```bash
python agent.py "What is the address of vitalik.eth?"
```

---

## 🧱 Architecture (truthful)

- **Blockscout MCP Server (Docker)** — serves **MCP JSON‑RPC** at `/mcp` (and optionally `/v1/...` REST if you enable `--rest`).
- **Agent (Python CLI)** — sends prompts to **ASI:One**, lets the model decide the tool sequence, executes **MCP `tools/call`**, feeds results back, and prints a **clean final answer**.
- **No REST endpoint from the agent**. It’s a CLI only.

### Why results are clean

The agent implements **hardened output control**:

- The model is asked to end the final answer with a sentinel `<<<END>>>`.
- A **sanitizer** trims at the sentinel and rejects known “garbage” signatures (e.g., random particle system dumps, editor configs).
- If garbage is detected, it reprompts once for a concise, clean answer.
- Logging shows **short “thinking” summaries** of each step instead of raw dumps.

---

## ✅ Prerequisites

- **Docker** installed and running.
- **Python 3.10+** with **venv**.
- An **ASI:One API key** (environment variable `ASI_ONE_API_KEY`).

Minimal Python deps:

- `requests`
- `python-dotenv`

Install via:

```bash
python -m pip install -r requirements.txt
```

(Or install those two packages manually.)

---

## 🔧 Environment

Create `.env` in the project root with:

```
ASI_ONE_API_KEY=sk-your-asi-key
# Optional override (defaults shown below)
BLOCKSCOUT_MCP_URL=http://localhost:8001/mcp
```

> The model name is set **inside `agent.py`** (`ASI_MODEL="asi1-extended"` in the current file). Change it there if you prefer `asi1-mini`.

---

## ▶️ Run the Blockscout MCP server

Start the official container in **HTTP MCP** mode on host port **8001**:

```bash
docker run --rm -p 8001:8000 ghcr.io/blockscout/mcp-server:latest \
  python -m blockscout_mcp_server --http --rest --http-host 0.0.0.0 --http-port 8000
```

- MCP JSON‑RPC endpoint will be at: `http://localhost:8001/mcp`
- `/v1/...` REST endpoints are also enabled by `--rest`, but **the agent talks to `/mcp`**.

Leave this terminal **running**.

---

## ▶️ Run the agent

In another terminal (with your virtualenv active):

```bash
python agent.py "What is the address of vitalik.eth?"
```

You should see logs like:

```
✅ MCP connected at http://localhost:8001/mcp. 18 tools available.
[thinking] received the request and prepared the tool environment

--- Step 1 ---
[thinking] deciding whether to call a tool or answer directly
[thinking] chose tool: get_address_by_ens_name — purpose: resolve ENS name to address
[thinking] with parameters: {"name":"vitalik.eth"}
[thinking] tool executed; resolved address = 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045

[answer]
0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045
```

The agent prints:

- **thinking**: short, readable step summaries (tool chosen, why, and high‑level result).
- **answer**: the clean final answer (sanitized; no random noise).

---

## 🛠️ How it works (accurate)

- Discovers available tools with `tools/list` from the MCP server on startup.
- Lets the LLM decide **if** and **which** tool to call. The agent does **not** hardcode keyword triggers.
- Accepts multiple JSON shapes from the LLM:
  - `{"tool":"...", "params": {...}}`
  - `{"tool":"...", "arguments": {...}}`
  - `{"name":"...", "params": {...}}`
  - `{"name":"...", "arguments": {...}}`
- Normalizes common parameter aliases (e.g., `ens_name → name` for `get_address_by_ens_name`).
- Calls `tools/call` over MCP JSON‑RPC and hands the result back to the model to decide next steps.
- Final answers are requested with a sentinel and sanitized before display.

**No raw chain‑of‑thought is printed.** The “thinking” lines are curated, high‑level summaries only.

---

## 🔬 Example commands you can run

```bash
# Simple ENS → address
python agent.py "What is the address of vitalik.eth?"

# NFTs with pagination
python agent.py "For punk6529.eth, get all NFTs held on Ethereum and summarize totals per collection."

# Transaction forensics
python agent.py "Analyze tx 0xf8a55721f7e2dcf85690aaf81519f7bc820bc58a878fa5f81b12aef5ccda0efb on Base (8453): decode input, summarize transfers and fees."

# Token symbol discovery + balances
python agent.py "On Optimism (10), find the OP token contract, then for barnbridge.eth list current balance and allowances."
```

For a longer test suite, see the “complex prompts” in your history; the agent supports multi‑step plans (resolve ENS → fetch transfers → aggregate, etc.).

---

## 🚑 Troubleshooting

**MCP `tools/list` 406 / 404**  
Make sure you are calling the MCP endpoint (`/mcp`) and sending:

```
Content-Type: application/json
Accept: application/json, text/event-stream
```

The agent already sets these headers. Use the Docker command above.

**Read timeouts from ASI**  
The agent uses `(connect=10s, read=45s)` timeouts. Network hiccups can cause read timeouts; just rerun the command. You can tweak timeouts in `agent.py` (`ASI_CONNECT_TIMEOUT` / `ASI_READ_TIMEOUT`).

**Random “particle.png/timeline” garbage in the output**  
The sanitizer will reject and reprompt. If you still see noise, update the `GARBAGE_PATTERNS` in `agent.py` with the new signature and rerun.

**PowerShell `curl` issues**  
On Windows, prefer `Invoke-WebRequest`/`Invoke-RestMethod` for testing endpoints, or use `curl.exe` explicitly. The agent itself does **not** require curl; it’s a CLI.

**Change model**  
Edit `ASI_MODEL` in `agent.py` (default in this repo is `asi1-extended`).

**Point to a public MCP**  
Set `BLOCKSCOUT_MCP_URL` in `.env` to the desired MCP JSON‑RPC endpoint (must speak MCP JSON‑RPC at `/mcp`).

---

## 📁 Project layout (key files)

```
agent.py          # The MCP-first CLI agent with hardened output control + think logs
requirements.txt  # Minimal Python dependencies
.env              # Your ASI_ONE_API_KEY and optional BLOCKSCOUT_MCP_URL
```

---

## 📄 License

MIT (or your preferred license).
