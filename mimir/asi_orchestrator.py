import os
import json
import requests
from typing import Dict, Any, List, Optional, Union
from decimal import Decimal, getcontext
from dotenv import load_dotenv

# --------------------------------------------------------------------------------------
# Load environment variables from .env automatically
# --------------------------------------------------------------------------------------
load_dotenv()

# Higher precision for big-number divisions (wei -> ETH)
getcontext().prec = 50

# Base URL and API key for ASI:One
ASI_BASE = os.getenv("ASI_BASE", "https://api.asi1.ai")
ASI_API_KEY = os.getenv("ASI_API_KEY")
ASI_MODEL = os.getenv("ASI_MODEL", "asi1-mini")  # correct model name

# Optional: log whether the API key is loaded (without revealing it)
if ASI_API_KEY:
    print("[asi_orchestrator] ✅ ASI_API_KEY loaded from .env")
else:
    print("[asi_orchestrator] ⚠️  ASI_API_KEY not found! Please set it in .env")

HEADERS = {
    "Authorization": f"Bearer {ASI_API_KEY}" if ASI_API_KEY else "",
    "Content-Type": "application/json",
}

# --------------------------------------------------------------------------------------
# ASI HTTP helper
# --------------------------------------------------------------------------------------
def _post_asi_chat(body: Dict[str, Any]) -> Dict[str, Any]:
    if not ASI_API_KEY:
        raise RuntimeError("ASI_API_KEY is not set")
    resp = requests.post(f"{ASI_BASE}/v1/chat/completions", json=body, headers=HEADERS, timeout=60)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(f"ASI error {resp.status_code}: {resp.text}") from e
    return resp.json()

# --------------------------------------------------------------------------------------
# Tool selection via ASI
# --------------------------------------------------------------------------------------
def choose_tool_with_asi(user_prompt: str, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Send the user's prompt + dynamic MCP tool schemas to ASI:One.
    Return {"name": <tool_name>, "arguments": {...}} or {"name": None}.
    """
    body = {
        "model": ASI_MODEL,
        "messages": [
            {"role": "system", "content": "You are a blockchain data assistant. Choose and call the best tool."},
            {"role": "user", "content": user_prompt}
        ],
        "tools": tools,
        "tool_choice": "auto"
    }
    data = _post_asi_chat(body)

    # OpenAI-compatible tool calling extraction
    for choice in data.get("choices", []):
        msg = choice.get("message", {}) or {}
        tc_list = msg.get("tool_calls") or []
        if tc_list:
            fn = tc_list[0].get("function", {})
            return {"name": fn.get("name"), "arguments": fn.get("arguments", {})}
        tc = msg.get("tool_call")
        if tc and tc.get("name"):
            return {"name": tc["name"], "arguments": tc.get("arguments", {})}

    return {"name": None, "arguments": {}}

# --------------------------------------------------------------------------------------
# Deterministic helpers to avoid hallucinations in formatting
# --------------------------------------------------------------------------------------
def _is_int_str(s: Union[str, int]) -> bool:
    if isinstance(s, int):
        return True
    if not isinstance(s, str):
        return False
    return s.isdigit()

def _format_eth_from_wei(wei_value: Union[str, int]) -> str:
    """Return a string with full-precision ETH, from wei."""
    if not _is_int_str(wei_value):
        return ""
    eth = (Decimal(str(wei_value)) / Decimal(10) ** 18)
    # Keep as many decimals as necessary; strip trailing zeros
    s = format(eth, 'f').rstrip('0').rstrip('.') if '.' in format(eth, 'f') else str(eth)
    return s

def _comma(n: Union[int, str]) -> str:
    try:
        return f"{int(n):,}"
    except Exception:
        return str(n)

def _detect_get_address_info(result: Dict[str, Any]) -> Optional[str]:
    """
    Deterministically produce a human-readable summary for get_address_info results
    without relying on the LLM. Returns None if the shape doesn't match.
    """
    try:
        data = result["data"]
        basic = data.get("basic_info", {})
        address = basic.get("hash")
        ens = basic.get("ens_domain_name")
        block = basic.get("block_number_balance_updated_at")
        coin_balance_wei = basic.get("coin_balance")
        is_contract = basic.get("is_contract")
        is_verified = basic.get("is_verified")
        proxy_type = basic.get("proxy_type")
        implementations = basic.get("implementations") or []
        tags = (data.get("metadata") or {}).get("tags") or []
        exchange_rate = basic.get("exchange_rate")  # may be None or string

        if not address or coin_balance_wei is None:
            return None

        # Convert wei -> ETH safely
        eth_str = _format_eth_from_wei(coin_balance_wei)
        wei_str = str(coin_balance_wei)
        addr_short = f"{address[:6]}...{address[-4:]}" if isinstance(address, str) else str(address)

        # Optional USD valuation if exchange_rate present and numeric-ish
        usd_line = ""
        try:
            if exchange_rate is not None and str(exchange_rate).replace('.', '', 1).isdigit():
                usd_val = (Decimal(eth_str) * Decimal(str(exchange_rate)))
                usd_line = f"\n- **Value (USD)**: ~${usd_val:,.2f} (rate: ${exchange_rate}/ETH)"
        except Exception:
            pass

        # Contract + impl hint
        impl_hint = ""
        if implementations:
            imp = implementations[0]
            impl_addr = imp.get("address_hash")
            impl_name = imp.get("name") or "Implementation"
            if impl_addr:
                impl_hint = f"\n- **Implementation**: {impl_name} (`{impl_addr[:6]}...{impl_addr[-4:]}`)"
            else:
                impl_hint = f"\n- **Implementation**: {impl_name}"

        # Tags (names only)
        tag_names = []
        for t in tags:
            name = t.get("name")
            if name:
                tag_names.append(name)
        tag_line = f"\n- **Tags**: " + ", ".join(tag_names) if tag_names else ""

        lines = [
            f"### Address Balance Summary",
            f"The address **{ens or addr_short}** (`{addr_short}`) holds **{eth_str} ETH**, last updated at block {_comma(block)}.",
            "",
            "---",
            "",
            "### Key Details",
            f"- **Address**: `{addr_short}`" + (f" ({ens})" if ens else ""),
            f"- **Balance**: {eth_str} ETH ({_comma(wei_str)} wei)",
        ]
        if is_contract is not None:
            lines.append(f"- **Address Type**: {'Contract' if is_contract else 'Externally Owned Account (EOA)'}")
        if is_verified is not None:
            lines.append(f"- **Verified**: {'Yes' if is_verified else 'No'}")
        if proxy_type:
            lines.append(f"- **Proxy Type**: {proxy_type}")
        if impl_hint:
            lines.append(impl_hint)
        if usd_line:
            lines.append(usd_line)
        if tag_line:
            lines.append(tag_line)

        lines += [
            "",
            "---",
            "",
            "The ETH value is computed from `coin_balance` (wei) using 18 decimals: **ETH = wei / 10^18**."
        ]
        return "\n".join(lines)
    except Exception:
        return None

# --------------------------------------------------------------------------------------
# Pretty formatting via deterministic pass + ASI fallback (general-purpose)
# --------------------------------------------------------------------------------------
def pretty_format_result(user_prompt: str, tool_name: str, tool_args: Dict[str, Any], raw_result: Dict[str, Any]) -> str:
    """
    First try to produce a deterministic, unit-safe summary for known tools.
    If not applicable, call ASI to transform the JSON into a concise Markdown answer.
    """

    # 1) Deterministic formatters for known tools
    if tool_name == "get_address_info":
        det = _detect_get_address_info(raw_result)
        if det:
            return det

    # 2) Fall back to ASI formatter for everything else
    raw_json = json.dumps(raw_result, ensure_ascii=False)
    if len(raw_json) > 120_000:
        raw_json = raw_json[:120_000] + "...(truncated)"

    system_instructions = (
        "You are a results formatter. Convert raw JSON outputs from blockchain tools into a concise, "
        "human-friendly Markdown response. STRICT RULES:\n"
        "1) Do NOT guess units or decimals. If a field is labeled 'wei', only convert to ETH if the conversion factor (1e18) is standard EVM behavior; otherwise show both original and converted.\n"
        "2) Use exact math; do not round unless asked. Prefer showing both precise and compact forms.\n"
        "3) Never invent fields or values. If a number cannot be derived unambiguously, show the raw number and label it clearly.\n"
        "4) Keep it brief: short summary + bullet points. Use thousands separators where appropriate.\n"
    )

    user_payload = (
        f"User prompt:\n{user_prompt}\n\n"
        f"Tool called: {tool_name}\n"
        f"Arguments: {json.dumps(tool_args, ensure_ascii=False)}\n\n"
        f"Raw JSON result:\n```json\n{raw_json}\n```"
    )

    body = {
        "model": ASI_MODEL,
        "messages": [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_payload}
        ],
        "temperature": 0.2,
        "max_tokens": 1000
    }

    data = _post_asi_chat(body)

    # Extract final message text
    for choice in data.get("choices", []):
        msg = choice.get("message", {}) or {}
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

    # Fallback: compact JSON slice
    return f"{tool_name}({tool_args}) → {raw_json[:1600]}" + ("…" if len(raw_json) > 1600 else "")
