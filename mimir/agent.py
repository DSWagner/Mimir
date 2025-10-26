#!/usr/bin/env python3
"""
MCP-first agent with HARDENED OUTPUT CONTROL + THINK LOGS (no raw dumps).

What this file does:
- Lets the LLM decide when/which MCP tools to call (no keyword forcing).
- Executes Blockscout MCP JSON-RPC tool calls and feeds results back to the LLM.
- Strong guardrails to keep final answers clean (sentinel, sanitizer, SSE watchdog).
- Replaces raw message dumps with concise, high-level THINK LOGS you can show to users.

Important policy note:
- We DO NOT print the model's literal chain-of-thought or raw responses.
- Instead we print brief summaries of each step that describe what happened
  (e.g., which tool was chosen and why at a high level) without exposing internal reasoning text.
"""

import json
import os
import re
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Config
# -----------------------------
ASI_API_KEY = os.getenv("ASI_ONE_API_KEY", "sk-REPLACE_ME")
ASI_ENDPOINT = "https://api.asi1.ai/v1/chat/completions"
ASI_MODEL = "asi1-extended"   # your preferred fast model
ASI_CONNECT_TIMEOUT = 10
ASI_READ_TIMEOUT = 45
ASI_TIMEOUT = (ASI_CONNECT_TIMEOUT, ASI_READ_TIMEOUT)

MCP_URL = os.getenv("BLOCKSCOUT_MCP_URL", "http://localhost:8001/mcp")
MCP_TIMEOUT = 60
MAX_TOOL_STEPS = 120

FINAL_SENTINEL = "<<<END>>>"

SESSION_MAP: Dict[str, str] = {}

# -----------------------------
# Think-log helpers (safe summaries)
# -----------------------------
def think(msg: str) -> None:
    """Print a concise, user-friendly reasoning summary (not raw model output)."""
    print(f"[thinking] {msg}")

def step_header(i: int) -> None:
    print(f"\n--- Step {i} ---")

def summarize_tool_purpose(name: str) -> str:
    """High-level purpose line per tool (for clean logs)."""
    purposes = {
        "__unlock_blockchain_analysis__": "initialize server guidance",
        "get_chains_list": "list supported chains",
        "get_address_by_ens_name": "resolve ENS name to address",
        "lookup_token_by_symbol": "find token addresses by symbol/name",
        "get_contract_abi": "fetch a contract ABI",
        "inspect_contract_code": "fetch verified contract source files",
        "get_address_info": "summarize address balances/tags/contracts",
        "get_tokens_by_address": "list ERC-20 holdings for an address",
        "get_latest_block": "fetch latest indexed block",
        "get_transactions_by_address": "list transactions in a time window",
        "get_token_transfers_by_address": "list ERC-20 transfers in a time window",
        "transaction_summary": "get human-readable tx summary",
        "nft_tokens_by_address": "list NFTs owned by an address",
        "get_block_info": "get detailed block info",
        "get_transaction_info": "get detailed transaction info",
        "get_transaction_logs": "get logs/events for a tx",
        "read_contract": "read a contract function",
        "direct_api_call": "call curated raw Blockscout endpoint",
    }
    return purposes.get(name, "perform on-chain lookup")

def summarize_tool_result(tool: str, payload: Dict[str, Any]) -> str:
    """Short, safe result summary (avoid raw dump)."""
    if not isinstance(payload, dict):
        return "received non-JSON response"

    if payload.get("error"):
        return "tool returned an error"

    # Common structured result field:
    data = payload.get("data")

    if tool == "get_address_by_ens_name" and isinstance(data, dict):
        addr = data.get("resolved_address")
        return f"resolved address = {addr}" if addr else "no address found"

    if tool == "get_latest_block" and isinstance(data, dict):
        number = data.get("number") or data.get("latest_block") or data.get("height")
        ts = data.get("timestamp")
        base = f"latest block = {number}" if number is not None else "latest block unknown"
        return f"{base}; timestamp = {ts}" if ts is not None else base

    if tool in ("get_tokens_by_address", "nft_tokens_by_address"):
        # Often paginated. We avoid listing all items; just count.
        count = None
        if isinstance(data, dict):
            items = data.get("items") or data.get("tokens")
            if isinstance(items, list):
                count = len(items)
        return f"received {count if count is not None else 'some'} items"

    if tool in ("get_transactions_by_address", "get_token_transfers_by_address"):
        count = None
        if isinstance(data, dict):
            items = data.get("items") or data.get("transactions") or data.get("transfers")
            if isinstance(items, list):
                count = len(items)
        return f"received {count if count is not None else 'some'} records"

    # Default: mention top-level keys only
    keys = ", ".join(list(payload.keys())[:6])
    return f"ok; keys: {keys}" if keys else "ok"

# -----------------------------
# Session handling
# -----------------------------
def get_session_id(conv_id: str) -> str:
    sid = SESSION_MAP.get(conv_id)
    if sid is None:
        sid = str(uuid.uuid4())
        SESSION_MAP[conv_id] = sid
    return sid

# -----------------------------
# LLM chat
# -----------------------------
def asi_chat(messages: List[Dict[str, str]], *, stream: bool = False) -> str:
    """LLM request helper. Finals are non-stream; tool-iteration is also non-stream here."""
    headers = {
        "Authorization": f"Bearer {ASI_API_KEY}",
        "x-session-id": get_session_id("default"),
        "Content-Type": "application/json",
    }
    payload = {"model": ASI_MODEL, "messages": messages, "stream": stream}

    r = requests.post(ASI_ENDPOINT, headers=headers, json=payload, timeout=ASI_TIMEOUT)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# -----------------------------
# MCP JSON-RPC client
# -----------------------------
class McpClient:
    def __init__(self, url: str, timeout: int = 60):
        self.url = url
        self.timeout = timeout
        self._id = 0
        self._headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Cache-Control": "no-store",
            "User-Agent": "Mimir-MCP-Agent/2.1",
        }

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    def _parse_sse(self, resp: requests.Response) -> Dict[str, Any]:
        last_obj: Optional[Dict[str, Any]] = None
        IDLE_MAX = 15
        last_activity = time.time()

        for line in resp.iter_lines(decode_unicode=True):
            if line:
                last_activity = time.time()
            if time.time() - last_activity > IDLE_MAX:
                break
            if not line:
                continue
            if line.startswith(":") or line.startswith("event:"):
                continue
            if line.startswith("data:"):
                data = line[len("data:"):].strip()
                if not data:
                    continue
                try:
                    obj = json.loads(data)
                    last_obj = obj
                    if isinstance(obj, dict) and ("result" in obj or "error" in obj):
                        break
                except Exception:
                    continue

        if last_obj is None:
            raise RuntimeError("Empty SSE stream from MCP server")
        if isinstance(last_obj, dict) and last_obj.get("error"):
            raise RuntimeError(f"MCP error: {last_obj['error']}")
        if isinstance(last_obj, dict) and "result" in last_obj:
            return last_obj["result"]
        return last_obj

    def rpc(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {"jsonrpc": "2.0", "id": self._next_id(), "method": method, "params": params or {}}
        with requests.post(self.url, headers=self._headers, json=payload, timeout=self.timeout, stream=True) as resp:
            resp.raise_for_status()
            ctype = resp.headers.get("Content-Type", "")
            if "text.event-stream" in ctype or "text/event-stream" in ctype:
                return self._parse_sse(resp)
            data = resp.json()
            if isinstance(data, dict) and data.get("error"):
                raise RuntimeError(f"MCP rpc error: {data['error']}")
            return data.get("result", {}) if isinstance(data, dict) else {}

    def tools_list(self) -> List[Dict[str, Any]]:
        result = self.rpc("tools/list", {})
        tools = result.get("tools", [])
        return tools if isinstance(tools, list) else []

    def tools_call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return self.rpc("tools/call", {"name": name, "arguments": arguments})

mcp = McpClient(MCP_URL, MCP_TIMEOUT)

# Discover tools (for display/validation only)
try:
    meta = mcp.tools_list()
    AVAILABLE_TOOLS = [t["name"] for t in meta if isinstance(t, dict) and "name" in t]
    print(f"✅ MCP connected at {MCP_URL}. {len(AVAILABLE_TOOLS)} tools available.")
except Exception as e:
    print(f"⚠️ tools/list failed: {e}")
    AVAILABLE_TOOLS = []

# -----------------------------
# Intent extraction
# -----------------------------
# Intent key mapping: short snake_case keys for LLM responses
INTENT_KEY_MAP = {
    'onchain': 'On-chain data analysis',
    'token': 'Token analysis',
    'defi': 'DeFi analysis',
    'security': 'Security',
    'other': 'Other',
}

# Reverse mapping for display
INTENT_DISPLAY_MAP = {v: k for k, v in INTENT_KEY_MAP.items()}


def extract_intent(text: str) -> Optional[str]:
    """
    Extract intent from LLM reply.
    Looks for intent in JSON tool call or standalone JSON.
    Returns the full intent name (e.g., 'On-chain data analysis') or None.
    """
    # Try to find JSON in the text
    try:
        # Try direct JSON parse first
        obj = json.loads(text.strip())
        if isinstance(obj, dict) and "intent" in obj:
            intent_key = obj["intent"]
            # Map short key to full name
            return INTENT_KEY_MAP.get(intent_key, intent_key)
    except Exception:
        pass

    # Fallback: find first JSON block
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e > s:
        try:
            snippet = text[s:e + 1]
            obj = json.loads(snippet)
            if isinstance(obj, dict) and "intent" in obj:
                intent_key = obj["intent"]
                # Map short key to full name
                return INTENT_KEY_MAP.get(intent_key, intent_key)
        except Exception:
            pass

    return None


# -----------------------------
# Tool-call parsing & normalization
# -----------------------------
def _normalize_toolcall_object(obj: Any) -> Optional[Dict[str, Any]]:
    """
    Accepts:
      {"tool": "...", "params": {...}, "intent": "..."}  # intent is optional
      {"tool": "...", "arguments": {...}}
      {"name": "...", "params": {...}}
      {"name": "...", "arguments": {...}}
    -> returns {"tool": name, "params": params, "intent": intent (optional)}
    """
    if not isinstance(obj, dict):
        return None

    tool_name = None
    params = None
    intent = None

    if "tool" in obj:
        tool_name = obj["tool"]
    elif "name" in obj:
        tool_name = obj["name"]

    if "params" in obj and isinstance(obj["params"], dict):
        params = obj["params"]
    elif "arguments" in obj and isinstance(obj["arguments"], dict):
        params = obj["arguments"]

    # Extract intent if present (optional field)
    if "intent" in obj and isinstance(obj["intent"], str):
        intent = obj["intent"]

    if tool_name and isinstance(params, dict):
        result = {"tool": tool_name, "params": params}
        if intent:
            result["intent"] = intent
        return result

    return None

def maybe_extract_tool_call(text: str) -> Optional[Dict[str, Any]]:
    # Try direct JSON first
    try:
        obj = json.loads(text)
        call = _normalize_toolcall_object(obj)
        if call:
            return call
    except Exception:
        pass

    # Fallback: find first JSON block
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e > s:
        try:
            snippet = text[s:e + 1]
            obj2 = json.loads(snippet)
            call = _normalize_toolcall_object(obj2)
            if call:
                return call
        except Exception:
            pass

    return None

def normalize_params(tool: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Surgical normalization (no user-prompt inference)."""
    p = dict(params or {})

    # get_address_by_ens_name expects {"name": "<ens>"}
    if tool == "get_address_by_ens_name":
        for alias in ("ens_name", "domain", "ens", "ensName"):
            if alias in p and "name" not in p:
                p["name"] = p.pop(alias)

    # read_contract sanity aliases
    if tool == "read_contract":
        if "fn" in p and "function_name" not in p:
            p["function_name"] = p.pop("fn")
        if "arguments" in p and "args" not in p and isinstance(p["arguments"], (list, tuple)):
            p["args"] = p.pop("arguments")

    return p

# -----------------------------
# Final answer hygiene
# -----------------------------
GARBAGE_PATTERNS = [
    r"\bparticle\.png\b",
    r"\bSpawn Shape\b",
    r"\bTint\b\s*-\s*",
    r"\bTransparency\b\s*-\s*",
    r"\bEmission\b\s*-\s*",
    r"\bRotation\b\s*-\s*",
    r"\bVelocity\b\s*-\s*",
    r"\btimelineCount\b",
    r"\bcolors\d+\b",
    r"\bscaling\d+\b",
]
GARBAGE_RE = re.compile("|".join(GARBAGE_PATTERNS), re.IGNORECASE | re.MULTILINE)

def sanitize_final(text: str) -> Optional[str]:
    """Cut at sentinel; reject if known garbage present; return clean final string or None to reprompt."""
    if FINAL_SENTINEL in text:
        text = text.split(FINAL_SENTINEL, 1)[0]

    text = text.strip().strip("`").strip()

    if GARBAGE_RE.search(text or ""):
        return None

    MAX_CHARS = 12000
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + " …"

    return text if text else None

FINAL_INSTRUCTION = (
    "Now produce ONLY the final user-facing answer, based on the prior specification. do not respond with code or explain why some tool failed, if so only state that some services were unavailable and state which tool was used for the source. Respond with a consize recap of the analysis and answer "
    f"and END with the token {FINAL_SENTINEL}. Do not add anything after the sentinel."
)

# -----------------------------
# System prompt builder
# -----------------------------
def build_system_prompt(intent: Optional[str] = None) -> str:
    """
    Build system prompt, optionally with intent-specific guidance.
    If intent is None, returns base prompt that asks LLM to provide intent.
    """
    base = (
        f"You are a blockchain expert. You can call Blockscout MCP tools via JSON-RPC at {MCP_URL}.\n"
        "Use your judgment to decide when tool calls are needed and which tools to use.\n"
    )

    if intent is None:
        # Initial prompt: ask LLM to classify intent using short keys
        intent_options = ", ".join([f'"{k}"' for k in INTENT_KEY_MAP.keys()])
        base += (
            "When you choose to call a tool, output exactly ONE JSON object with keys `tool`, `params`, and `intent`.\n"
            f"The `intent` field must be ONE of: {intent_options}\n"
            "Intent meanings:\n"
            "- \"onchain\": On-chain data analysis (addresses, transactions, blocks, contracts)\n"
            "- \"token\": Token analysis (ERC-20, ERC-721, NFTs, token metadata)\n"
            "- \"defi\": DeFi protocol analysis (swaps, liquidity, staking)\n"
            "- \"security\": Security analysis (audits, vulnerabilities)\n"
            "- \"other\": General blockchain queries\n"
            "Example: {\"tool\":\"get_address_by_ens_name\",\"params\":{\"name\":\"vitalik.eth\"},\"intent\":\"onchain\"}\n"
        )
    else:
        # Subsequent calls: intent already classified, just ask for tool/params
        base += (
            "When you choose to call a tool, output exactly ONE JSON object with keys `tool` and `params`.\n"
            "Example: {\"tool\":\"get_address_by_ens_name\",\"params\":{\"name\":\"vitalik.eth\"}}\n"
        )

        # Add intent-specific guidance
        intent_guides = {
            'On-chain data analysis': (
                "\n=== PRIMARY FOCUS: On-Chain Data Analysis ===\n"
                "You are analyzing blockchain data such as addresses, transactions, blocks, and contracts.\n"
                "Prioritize accuracy and completeness when retrieving on-chain information.\n"
                "Key patterns:\n"
                "- For addresses: use get_address_info, get_tokens_by_address, nft_tokens_by_address\n"
                "- For transactions: use get_transaction_info, get_transaction_logs, transaction_summary\n"
                "- For contracts: use get_contract_abi, inspect_contract_code, read_contract\n"
                "- For ENS: use get_address_by_ens_name first\n"
            ),
            'Token analysis': (
                "\n=== PRIMARY FOCUS: Token Analysis ===\n"
                "You are analyzing tokens (ERC-20, ERC-721, ERC-1155).\n"
                "Focus on token metadata, supply, distribution, transfers, and holder information.\n"
                "Key patterns:\n"
                "- Use lookup_token_by_symbol to find token contracts\n"
                "- Use read_contract for metadata (name, symbol, totalSupply, decimals)\n"
                "- Use get_token_transfers_by_address for transfer history\n"
                "- Use get_tokens_by_address for holdings\n"
                "Structure the resposne with a high level summary of the token and a detailed analysis of the token\n"
                "Provide information about which chains and networks the token is related to\n"
                "Provide a rich and insightful view of the token and related topics\n"
            ),
            'DeFi analysis': (
                "\n=== PRIMARY FOCUS: DeFi Protocol Analysis ===\n"
                "You are analyzing DeFi protocols, liquidity pools, or decentralized exchange activity.\n"
                "Look for contract interactions, token flows, and protocol-specific events.\n"
                "Key patterns:\n"
                "- Examine transaction logs for swap/transfer events\n"
                "- Use read_contract to query pool states\n"
                "- Track token flows using get_token_transfers_by_address\n"
                "- Analyze contract interactions and event logs\n"
                "Structure the resposne with a high level summary of the token and a detailed analysis of the token\n"
                "Provide information about which chains and networks the token is related to\n"
                "Provide a rich and insightful view of the token and related topics\n"
            ),
            'Security': (
                "\n=== PRIMARY FOCUS: Security Analysis ===\n"
                "You are performing security-related analysis on contracts or transactions.\n"
                "Acces and examine contract code via MCP, transaction patterns, and potential vulnerabilities carefully.\n"
                "Key patterns:\n"
                "- Use inspect_contract_code to review source code\n"
                "- Use get_transaction_logs to examine event patterns\n"
                "- Check contract verification status\n"
                "- Analyze transaction flows for suspicious patterns\n"
                "- Access the raw code a perform a simplyfied static anlysis of the code itself\n"
                "IMPORTANT: Provide informational analysis only; do not make definitive security claims.\n"
            ),
            'Other': (
                "\n=== PRIMARY FOCUS: General Blockchain Query ===\n"
                "You are answering a general blockchain question.\n"
                "Adapt your approach based on what the user is asking.\n"
                "Provide a high level structured anyalysis and expand on related fact about the query\n"
                "Always include a short summary of queried topic and expand on it\n"
                "Provide information about which chains and networks the query is related to\n"
                "Provide a rich and insightful view of the query and related topics\n"
            ),
        }

        base += intent_guides.get(intent, intent_guides['Other'])

    base += (
        "\nAfter a tool result is returned to you, continue reasoning and decide the next step (another tool call or a final answer).\n"
        "Do not mention or rely on any external agent marketplace; focus on the Blockscout MCP tools only.\n"
        f"When delivering the FINAL ANSWER, end your message with the exact token {FINAL_SENTINEL}\n"
        "Available tools: " + ", ".join(AVAILABLE_TOOLS) + "\n"
    )

    return base


# Legacy static prompt (kept for reference)
SYSTEM_PROMPT = build_system_prompt(intent=None)

# -----------------------------
# Main loop
# -----------------------------
def run(user_prompt: str) -> None:
    think("received the request and prepared the tool environment")

    # Start with base prompt (asks for intent on first call)
    current_intent = None
    conv: List[Dict[str, str]] = [
        {"role": "system", "content": build_system_prompt(intent=None)},
        {"role": "user", "content": user_prompt},
    ]

    for step in range(1, MAX_TOOL_STEPS + 1):
        step_header(step)
        think("deciding whether to call a tool or answer directly")
        reply = asi_chat(conv, stream=False).strip()

        # Extract intent from first response if not already set
        if current_intent is None:
            extracted_intent = extract_intent(reply)
            if extracted_intent:
                current_intent = extracted_intent
                think(f"detected intent: {current_intent}")
                # Update system message with intent-specific guidance
                conv[0] = {"role": "system", "content": build_system_prompt(intent=current_intent)}

        tool_call = maybe_extract_tool_call(reply)
        if tool_call is None:
            think("no explicit tool call detected; asking the model for a clean final answer with sentinel")
            conv2 = conv + [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": build_system_prompt(intent=current_intent)+FINAL_INSTRUCTION},
            ]
            final_text = asi_chat(conv2, stream=False)
            clean = sanitize_final(final_text)
            if clean is None:
                think("final answer contained unrelated noise; reprompting once for a concise clean answer")
                conv3 = conv2 + [{"role": "user", "content": "Return ONLY the concise final answer. End with " + FINAL_SENTINEL}]
                final_text = asi_chat(conv3, stream=False)
                clean = sanitize_final(final_text)

            if clean is None:
                think("could not obtain a clean final answer; stopping")
                print("❌ Aborted: could not produce a clean final answer.")
                return

            print("\n[answer]")
            print(clean)
            return

        # Tool call path:
        tool_name = tool_call["tool"]
        params = normalize_params(tool_name, tool_call["params"])

        # Log the intention in a concise way
        think(f"chose tool: {tool_name} — purpose: {summarize_tool_purpose(tool_name)}")
        # Brief param echo (safe)
        think(f"with parameters: {json.dumps(params, separators=(',',':'))}")

        if AVAILABLE_TOOLS and tool_name not in AVAILABLE_TOOLS:
            payload = {"error": f"Unsupported MCP tool '{tool_name}'. Available: {AVAILABLE_TOOLS}"}
            think("tool not recognized by the connected MCP server")
        else:
            try:
                payload = mcp.tools_call(tool_name, params)
                think(f"payload: {payload}")
                think(f"tool executed; {summarize_tool_result(tool_name, payload)}")
            except Exception as exc:
                payload = {"error": f"MCP tools/call failed: {exc}"}
                think("tool execution failed")

        # Feed call + result back so the model can decide next step
        conv.append({"role": "assistant", "content": json.dumps({"tool": tool_name, "params": params}, separators=(",", ":"))})
        conv.append({
            "role": "user",
            "content": (
                f"TOOL_RESULT {tool_name} with params {json.dumps(params, separators=(',',':'))}:\n"
                + "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"
            ),
        })

        time.sleep(0.2)

    think("reached max tool steps without a final answer")
    print("⚠️ Reached max steps without a final answer.")

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python agent.py \"your question\"")
        sys.exit(1)
    prompt = " ".join(sys.argv[1:]).strip()
    run(prompt)
