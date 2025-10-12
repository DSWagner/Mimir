# agent.py
from datetime import datetime, timezone
from uuid import uuid4
import json
import re
from typing import Dict

from uagents import Agent, Context, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatMessage, ChatAcknowledgement, TextContent, chat_protocol_spec
)

from mcp import tools_list, tools_call, unwrap_content, mcp_health
from chains import select_chain_id, chain_name_by_id, get_chains
from utils import is_hex_address, trim_addr, fmt_eth, maybe_usd, pretty

def format_balance_response(raw: dict, chain_name: str, chain_id: str) -> str:
    """
    Format the MCP get_address_info / balance response nicely for chat output.
    Supports both {data:{...}} and flat {...} response structures.
    """

    # --- Normalize response structure ---
    if "data" in raw and isinstance(raw["data"], dict):
        data = raw["data"]
    elif "result" in raw and isinstance(raw["result"], dict):
        data = raw["result"]
    else:
        data = raw  # fallback if Blockscout MCP returned flat data

    # defensive extraction
    basic_info = data.get("basic_info") or data
    meta = data.get("metadata", {}) or {}
    tags = meta.get("tags", [])

    # safely handle missing fields
    address = basic_info.get("hash", "N/A")
    ens = basic_info.get("ens_domain_name", "N/A")
    balance_raw = basic_info.get("coin_balance", "0")
    try:
        balance_eth = int(balance_raw) / 1e18
    except Exception:
        balance_eth = 0.0

    exchange_rate = float(basic_info.get("exchange_rate", 0))
    usd_val = balance_eth * exchange_rate

    tag_lines = [f"• {t.get('name','')} ({t.get('tagType','')})" for t in tags] or ["• None"]
    tag_text = "\n".join(tag_lines)

    lines = [
        f"🪙  Balance Information — {chain_name} (chain_id: {chain_id})\n",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n",
        f"👤  ENS Name: {ens}\n",
        f"🏷️  Address: {address}\n",
        "",
        f"💰  Balance: {balance_eth:.8f} ETH\n",
        f"💵  (~${usd_val:,.2f} USD at ${exchange_rate}/ETH)\n",
        f"📦  Updated at Block: {basic_info.get('block_number_balance_updated_at', 'N/A')}\n",
        "\n",
        "───────────────────────────────────────────────\n",
        "📜  Account Details\n",
        f"• Verified: {'✅' if basic_info.get('is_verified') else '❌'}\n",
        f"• Smart Contract: {'✅' if basic_info.get('is_contract') else '❌'}\n",
        f"• Proxy Type: {basic_info.get('proxy_type', 'None')}\n",
        f"• Reputation: {basic_info.get('reputation', 'N/A')}\n",
        f"• Has Tokens: {'✅' if basic_info.get('has_tokens') else '❌'}\n",
        f"• Has Logs: {'✅' if basic_info.get('has_logs') else '❌'}\n",
        f"• Has Token Transfers: {'✅' if basic_info.get('has_token_transfers') else '❌'}\n",
        f"• Created by: {basic_info.get('creator_address_hash', 'N/A')}\n",
        f"• Creator Tx: {basic_info.get('creation_transaction_hash', 'N/A')}\n",
        "\n",
        "───────────────────────────────────────────────\n",
        "🏷️  Tags:\n",
        tag_text,
        "\n───────────────────────────────────────────────\n",
        "🔍  Try next:\n",
        f"→ `@mimir txs {ens or address} {chain_name.lower()} 5`   (recent transactions)\n",
        f"→ `@mimir call get_tokens_by_address {{\"chain_id\":\"{chain_id}\",\"address\":\"{address}\"}}`\n"
    ]

    return "\n".join(lines)


# ---- simple pagination memory (per sender) for txs "more" ----
_TXS_STATE: Dict[str, Dict[str, str]] = {}

# ---- Agent bootstrap ----
try:
    agent  # type: ignore  # noqa: F821
except NameError:
    agent = Agent(name="mimir", mailbox=True, publish_agent_details=True)

@agent.on_event("startup")
async def startup(ctx: Context):
    ctx.logger.info(f"Hello, I'm agent mimir and my address is {agent.address}.")
    ctx.logger.info("Using Blockscout MCP: https://mcp.blockscout.com/mcp")

def extract_text(msg: ChatMessage) -> str:
    return " ".join(c.text for c in msg.content if isinstance(c, TextContent)).strip()

def unwrap_data(payload: dict):
    if isinstance(payload, dict) and isinstance(payload.get("data"), (dict, list)):
        return payload["data"]
    return payload

# ---- Chat protocol ----
chat_proto = Protocol(spec=chat_protocol_spec)

@chat_proto.on_message(ChatMessage)
async def on_chat_message(ctx: Context, sender: str, msg: ChatMessage):
    # Ack for ASI chat
    await ctx.send(sender, ChatAcknowledgement(
        timestamp=datetime.now(timezone.utc),
        acknowledged_msg_id=msg.msg_id,
    ))

    text = extract_text(msg)
    low = text.lower().strip()

    # status
    if low in {"hi", "hello", "ping", "status"}:
        ok, body, code = mcp_health()
        reply = f"mimir is online. MCP is {'reachable ✅' if ok else 'unreachable ❌'}. HTTP {code} {body}"
        await ctx.send(sender, ChatMessage(timestamp=datetime.now(timezone.utc), msg_id=uuid4(),
            content=[TextContent(type="text", text=reply)]))
        return

    # tools
    if low.startswith("tools"):
        ok, res = tools_list()
        if ok and isinstance(res, dict) and "tools" in res:
            names = [t.get("name", "?") for t in res["tools"]]
            reply = "Available MCP tools:\n- " + "\n- ".join(names)
        else:
            reply = f"Could not list tools: {pretty(res)}"
        await ctx.send(sender, ChatMessage(timestamp=datetime.now(timezone.utc), msg_id=uuid4(),
            content=[TextContent(type="text", text=reply)]))
        return

    # chains
    if low.startswith("chains"):
        ok, chains = get_chains()
        if ok:
            lines = []
            for ch in chains:  # type: ignore
                cid = ch.get("chain_id")
                nm = ch.get("name") or ch.get("title") or ch.get("chain") or ch.get("slug")
                lines.append(f"{cid}: {nm}")
            reply = "Chains available: " + " ".join(lines) if lines else "No chains."
        else:
            reply = f"Failed to get chains: {chains}"
        await ctx.send(sender, ChatMessage(timestamp=datetime.now(timezone.utc), msg_id=uuid4(),
            content=[TextContent(type="text", text=reply)]))
        return

    # call <tool> <json-args>
    m = re.match(r"^\s*call\s+([A-Za-z0-9_:-]+)\s+(.*)$", text, flags=re.IGNORECASE)
    if m:
        tool = m.group(1)
        args_text = m.group(2).strip()
        try:
            args = json.loads(args_text) if args_text else {}
        except Exception as e:
            reply = f"Invalid JSON args: {e}"
        else:
            ok, res = tools_call(tool, args)
            payload = unwrap_content(res) if ok else res
            reply = pretty(payload) if ok else f"{tool} error: {pretty(payload)}"
        await ctx.send(sender, ChatMessage(timestamp=datetime.now(timezone.utc), msg_id=uuid4(),
            content=[TextContent(type="text", text=reply)]))
        return

    # balance <addr_or_ENS> [chain]
    m = re.match(r"^\s*balance\s+(\S+)(?:\s+(\S+))?\s*$", text, flags=re.IGNORECASE)
    if m:
        addr_or_ens = m.group(1)
        chain_token = m.group(2)

        ok_cid, cid_or_err = select_chain_id(chain_token)
        if not ok_cid:
            reply = str(cid_or_err)
        else:
            chain_id_str = str(cid_or_err)

            # Resolve ENS if needed
            if not is_hex_address(addr_or_ens):
                ok_res, ens_res = tools_call("get_address_by_ens_name", {"name": addr_or_ens, "chain_id": chain_id_str})
                if not ok_res:
                    reply = f"ENS resolution error: {pretty(ens_res)}"
                    await ctx.send(sender, ChatMessage(timestamp=datetime.now(timezone.utc), msg_id=uuid4(),
                        content=[TextContent(type="text", text=reply)]))
                    return
                u = unwrap_content(ens_res)
                address = (u.get("data") or {}).get("resolved_address") or u.get("address")
                if not address:
                    reply = f"ENS resolution returned unexpected payload: {pretty(u)}"
                    await ctx.send(sender, ChatMessage(timestamp=datetime.now(timezone.utc), msg_id=uuid4(),
                        content=[TextContent(type="text", text=reply)]))
                    return
            else:
                address = addr_or_ens

            # Prefer get_address_info; fallbacks
            for tool in ("get_address_info", "get_balance", "account_balance"):
                ok, res = tools_call(tool, {"chain_id": chain_id_str, "address": address})
                if ok:
                    data = unwrap_content(res)
                    data = data.get("data", data)
                    chain_name = chain_name_by_id(chain_id_str)

                    if isinstance(data, dict) and isinstance(data.get("basic_info"), dict):
                        b = data["basic_info"]
                        ens = b.get("ens_domain_name") or (addr_or_ens if addr_or_ens.endswith(".eth") else "(none)")
                        bal_wei = b.get("coin_balance")
                        px = b.get("exchange_rate")
                        tags = []
                        mtd = data.get("metadata") or {}
                        for t in (mtd.get("tags") or []):
                            nm = t.get("name")
                            if nm:
                                tags.append(nm)
                        tag_str = f" • tags: {', '.join(tags)}" if tags else ""
                        reply = format_balance_response(data, chain_name, chain_id_str)
                    else:
                        reply = f"{chain_name} • {trim_addr(address)}\n{pretty(data)}"
                    break
            else:
                reply = "No working balance tool found (tried get_address_info/get_balance/account_balance)."

        await ctx.send(sender, ChatMessage(timestamp=datetime.now(timezone.utc), msg_id=uuid4(),
            content=[TextContent(type="text", text=reply)]))
        return

    # txs <addr_or_ENS> [chain] [limit]
    m = re.match(r"^\s*txs\s+(\S+)(?:\s+(\S+))?(?:\s+(\d+))?\s*$", text, flags=re.IGNORECASE)
    if m:
        addr_or_ens = m.group(1)
        chain_token = m.group(2)
        limit = int(m.group(3)) if m.group(3) else 5

        ok_cid, cid_or_err = select_chain_id(chain_token)
        if not ok_cid:
            reply = str(cid_or_err)
        else:
            chain_id_str = str(cid_or_err)

            # Resolve ENS
            if not is_hex_address(addr_or_ens):
                ok_res, ens_res = tools_call("get_address_by_ens_name", {"name": addr_or_ens, "chain_id": chain_id_str})
                if not ok_res:
                    reply = f"ENS resolution error: {pretty(ens_res)}"
                    await ctx.send(sender, ChatMessage(timestamp=datetime.now(timezone.utc), msg_id=uuid4(),
                        content=[TextContent(type="text", text=reply)]))
                    return
                u = unwrap_content(ens_res)
                address = (u.get("data") or {}).get("resolved_address") or u.get("address")
                if not address:
                    reply = f"ENS resolution returned unexpected payload: {pretty(u)}"
                    await ctx.send(sender, ChatMessage(timestamp=datetime.now(timezone.utc), msg_id=uuid4(),
                        content=[TextContent(type="text", text=reply)]))
                    return
            else:
                address = addr_or_ens

            ok, res = tools_call("get_transactions_by_address",
                                 {"chain_id": chain_id_str, "address": address, "limit": limit})
            if not ok:
                reply = f"txs error: {pretty(res)}"
            else:
                body = unwrap_content(res)
                data = body.get("data", body)
                if not isinstance(data, list) or not data:
                    reply = f"{chain_name_by_id(chain_id_str)} • {trim_addr(address)}\nNo transactions found."
                else:
                    lines = []
                    for tx in data[:limit]:
                        ts = (tx.get("timestamp") or "").replace("T", " ").replace("Z", " UTC")
                        frm = trim_addr(tx.get("from",""))
                        to = trim_addr(tx.get("to",""))
                        val = fmt_eth(tx.get("value"))
                        fee = fmt_eth(tx.get("fee"))
                        method = tx.get("method") or "-"
                        h = tx.get("hash","")
                        lines.append(f"{ts} • {frm} → {to} • {val} • fee {fee} • {method} • {h[:10]}…")

                    # pagination
                    cursor = None
                    pag = (body.get("pagination") or {})
                    nxt = (pag.get("next_call") or {}).get("params") or {}
                    if nxt:
                        cursor = nxt.get("cursor")
                    if cursor:
                        _TXS_STATE[sender] = {
                            "chain_id": chain_id_str,
                            "address": address,
                            "cursor": cursor,
                        }
                        tail = "\n(more available — send: more)"
                    else:
                        tail = ""

                    reply = (
                        f"{chain_name_by_id(chain_id_str)} • {trim_addr(address)} • latest {min(limit,len(data))} txs\n"
                        + "\n".join(lines) + tail
                    )

        await ctx.send(sender, ChatMessage(timestamp=datetime.now(timezone.utc), msg_id=uuid4(),
            content=[TextContent(type="text", text=reply)]))
        return

    # more -> continue last txs page
    if low == "more":
        state = _TXS_STATE.get(sender)
        if not state:
            reply = "No pending pagination. Run a txs query first."
        else:
            args = {
                "chain_id": state["chain_id"],
                "address": state["address"],
                "cursor": state["cursor"],
            }
            ok, res = tools_call("get_transactions_by_address", args)
            if not ok:
                reply = f"txs error: {pretty(res)}"
            else:
                body = unwrap_content(res)
                data = body.get("data", body)
                if isinstance(data, list) and data:
                    lines = []
                    for tx in data:
                        ts = (tx.get("timestamp") or "").replace("T", " ").replace("Z", " UTC")
                        frm = trim_addr(tx.get("from",""))
                        to = trim_addr(tx.get("to",""))
                        val = fmt_eth(tx.get("value"))
                        fee = fmt_eth(tx.get("fee"))
                        method = tx.get("method") or "-"
                        h = tx.get("hash","")
                        lines.append(f"{ts} • {frm} → {to} • {val} • fee {fee} • {method} • {h[:10]}…")

                    cursor = None
                    pag = (body.get("pagination") or {})
                    nxt = (pag.get("next_call") or {}).get("params") or {}
                    if nxt:
                        cursor = nxt.get("cursor")
                    if cursor:
                        _TXS_STATE[sender]["cursor"] = cursor
                        tail = "\n(more available — send: more)"
                    else:
                        _TXS_STATE.pop(sender, None)
                        tail = ""

                    reply = "\n".join(lines) + tail
                else:
                    _TXS_STATE.pop(sender, None)
                    reply = "No more results."
        await ctx.send(sender, ChatMessage(timestamp=datetime.now(timezone.utc), msg_id=uuid4(),
            content=[TextContent(type="text", text=reply)]))
        return

    # help
    help_text = (
        "mimir is online.\n"
        "• tools — list MCP tools\n"
        "• chains — list available chains (name & id)\n"
        "• balance <address_or_ENS> [chain]\n"
        "• txs <address_or_ENS> [chain] [limit]\n"
        "• more — next page for the last txs query\n"
        "• call <tool_name> <json-args> (advanced)"
    )
    await ctx.send(sender, ChatMessage(timestamp=datetime.now(timezone.utc), msg_id=uuid4(),
        content=[TextContent(type="text", text=help_text)]))

@chat_proto.on_message(ChatAcknowledgement)
async def on_ack(_ctx: Context, _sender: str, _msg: ChatAcknowledgement):
    pass

agent.include(chat_proto, publish_manifest=True)

if __name__ == "__main__":
    agent.run()