# Mimir MCP Agent — Local Setup Guide

This repo wires up a local **uAgents**-based agent that talks to the **Blockscout MCP server** and uses **ASI:One** to (1) pick the right MCP tool and (2) prettify raw results into concise, human‑readable Markdown.

> TL;DR: You’ll run a Dockerized MCP server on port **8001**, start the Python agent on port **8000**, then send prompts via a simple REST call.

---

## 🧱 Architecture (high level)

- **Blockscout MCP Server (Docker)** → exposes blockchain tools over REST (we’ll use `GET /v1/...` endpoints)
- **Agent (Python/uAgents)** → receives your prompts, asks **ASI:One** which tool to call, executes it on the MCP server, then pretty‑formats the result
- **You (PowerShell)** → send a prompt to `http://127.0.0.1:8000/chat`

Ports:

- MCP server → **8001** (container `8000` mapped to host `8001`)
- Agent → **8000**

> ⚠️ **Important:** Keep **8001** for the MCP server to avoid clashing with the agent’s port (**8000**).

---

## ✅ Prerequisites

- **Windows** + **PowerShell** (examples use PS)
- **Docker** installed and running
- **Python 3.10+** with **venv**
- An **ASI API key** (for ASI:One) — this is required for tool selection & pretty summaries

---

## 1) Clone the repo

```powershell
git clone <your-repo-url> mimir
cd .\mimir
```

> The rest of the steps assume your current directory is the repo root (`.\mimir`).

---

## 2) Create & activate a virtual environment **inside `./mimir/`**

```powershell
# From .\mimir
py -m venv .\venv
.\venv\Scripts\Activate.ps1
```

Your prompt should now display `(venv)` at the beginning.

---

## 3) Install Python dependencies

We’ll install from the provided requirements file located at `./mimir/requirements.txt`:

```powershell
(venv) PS .\mimir> python -m pip install -r .\mimir\requirements.txt
```

If you’re already inside the `mimir` folder, you can also use:

```powershell
(venv) PS .\mimir> python -m pip install -r .\requirements.txt
```

> Tip: A minimal dependency set for this project is typically: `uagents`, `requests`, `python-dotenv`.

---

## 4) Configure your `.env`

Create a file named `.env` in the repo root (`.\mimir\.env`) with your ASI key:

```
ASI_API_KEY=your_asi_api_key_here
# Optional override (default is asi1-mini)
ASI_MODEL=asi1-mini
# MCP base URL (defaults to http://localhost:8001)
MCP_BASE_URL=http://localhost:8001
```

The agent loads this automatically (via `python-dotenv`).

---

## 5) Start the Blockscout MCP server (Docker)

Open a **new** PowerShell terminal (e.g., in Cursor IDE) and run:

```powershell
docker run --rm -p 8001:8000 ghcr.io/blockscout/mcp-server:latest python -m blockscout_mcp_server --http --rest --http-host 0.0.0.0
```

- Leave the **first port as 8001** (host:container is `8001:8000`) so it **doesn’t collide** with the agent (which runs on **8000**).
- You can sanity-check it with:
  ```powershell
  Invoke-RestMethod http://localhost:8001/v1/tools -Method GET | ConvertTo-Json -Depth 4
  ```

Leave this terminal **open** — the server must keep running.

---

## 6) Start the Agent

Back in your **venv terminal** (the one with `(venv)` in the prompt):

```powershell
(venv) PS .\mimir> python .\agent.py
```

You should see logs like:

```
[blockscout_mcp_agent]: Starting server on http://0.0.0.0:8000
[blockscout_mcp_agent]: Manifest published successfully: chat
[asi_orchestrator] ✅ ASI_API_KEY loaded from .env
```

> The agent automatically performs a one‑time “unlock” call if the MCP server requires it.

---

## 7) Send a test prompt

In **another** PowerShell terminal (while both the Docker container and the agent are running), send a prompt to the agent:

```powershell
$body = @{ text = "What is the balance on address 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045 on Ethereum chain?" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/chat" -Method POST -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 8
```

You should receive a human‑friendly Markdown summary (the agent uses ASI to format the raw JSON).

---

## 🔍 Troubleshooting

- **`ASI_API_KEY is not set`**  
  Ensure `.env` exists in the repo root and contains your key. Restart the agent after adding it.

- **`404 Not Found for url: https://api.asi1.ai/v1/chat/completions`**  
  Check that your model name is `asi1-mini` (with the `1`). You can set `ASI_MODEL=asi1-mini` in `.env`.

- **`404 Not Found` on `/tools/call`**  
  Your MCP build uses **per‑tool GET endpoints** (`/v1/get_address_info?…`) and **not** `POST /v1/tools/call`. The client code already handles this automatically.

- **Port conflicts**  
  Keep MCP on **8001** and the agent on **8000** as shown above.

- **Verify MCP server is reachable**
  ```powershell
  Invoke-RestMethod http://localhost:8001/v1/tools -Method GET
  ```

---

## 🧯 Notes

- The agent exposes a REST route at `POST /chat` that accepts `{ "text": "..." }` and returns `{ "text": "..." }`.
- The agent discovers MCP tools, orchestrates tool selection via ASI, and pretty‑formats results. It also auto‑calls the unlock tool once per run where required.
- This setup is **local‑first** — nothing is deployed publicly unless you choose to.

---

## 📄 License

MIT (or your preferred license). Update this section to match your project.
