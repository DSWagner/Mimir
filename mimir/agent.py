import os
import json
from typing import Dict, Any, Optional, List

from uagents import Agent, Context, Protocol, Model

from mcp_client import MCPClient
from asi_orchestrator import choose_tool_with_asi, pretty_format_result

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
AGENT_SEED = os.getenv("AGENT_SEED", "blockscout-mcp-auto-seed")
# IMPORTANT: this should point to your Dockerized MCP server (you mapped 8001:8000)
MCP_BASE_URL = os.getenv("MCP_BASE_URL", "http://localhost:8001")

# --------------------------------------------------------------------------------------
# Protocol models
# --------------------------------------------------------------------------------------
class ChatMessage(Model):
    text: str  # user message

class ChatResponse(Model):
    text: str  # agent reply

# REST models (for a simple REST endpoint that mirrors the chat protocol)
class RestIn(Model):
    text: str

class RestOut(Model):
    text: str

# --------------------------------------------------------------------------------------
# Agent + MCP client
# --------------------------------------------------------------------------------------
chat = Protocol(name="chat", version="1.0")
agent = Agent(name="blockscout_mcp_agent", seed=AGENT_SEED)
mcp = MCPClient(base_url=MCP_BASE_URL)

_cached_asi_tools: Optional[List[Dict[str, Any]]] = None
_unlock_done: bool = False  # <--- global flag instead of ctx.state

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _mcp_to_asi_tools(mcp_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert MCP tool descriptions to ASI tool schemas:
    MCP uses e.g. {name, description, input_schema, ...}
    ASI expects {name, description, parameters}
    """
    asi_tools: List[Dict[str, Any]] = []
    for t in mcp_tools:
        name = t.get("name")
        if not name:
            continue
        asi_tools.append({
            "name": name,
            "description": t.get("description", ""),
            "parameters": t.get("input_schema", {"type": "object", "properties": {}})
        })
    return asi_tools


def _ensure_unlock(ctx: Context):
    """Call the unlock tool once per process (safe no-op if not required)."""
    global _unlock_done
    if _unlock_done:
        return
    try:
        mcp.call_tool("__unlock_blockchain_analysis__", {})
        ctx.logger.info("Unlock tool executed.")
    except Exception as e:
        ctx.logger.warning(f"Unlock tool failed or not required: {e}")
    finally:
        _unlock_done = True


def _mcp_chat_flow(ctx: Context, prompt: str) -> str:
    """
    Shared logic used by both the chat protocol handler and the REST endpoint:
    - Discover MCP tools (cached)
    - (Optional) Unlock
    - Ask ASI to choose the best tool + args
    - Call the MCP tool
    - Pretty-format the result with ASI
    """
    global _cached_asi_tools

    # 1) Discover tools once (cached)
    if _cached_asi_tools is None:
        mcp_tools = mcp.list_tools()
        _cached_asi_tools = _mcp_to_asi_tools(mcp_tools)
        ctx.logger.info(f"Discovered {len(_cached_asi_tools)} MCP tools.")

    # 2) Ensure unlock has been attempted
    _ensure_unlock(ctx)

    # 3) Ask ASI which tool to call
    choice = choose_tool_with_asi(prompt, _cached_asi_tools)
    name, args = choice.get("name"), choice.get("arguments", {})

    if not name:
        return "I couldn't pick a suitable tool for that request."

    # 4) If ASI returned arguments as a JSON-encoded string, try to parse
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            pass

    # 5) Call MCP tool
    result = mcp.call_tool(name, args)

    # 6) Pretty-format via ASI (fallback to compact JSON if anything fails)
    try:
        pretty = pretty_format_result(
            user_prompt=prompt,
            tool_name=name,
            tool_args=args,
            raw_result=result
        )
        return pretty
    except Exception as e:
        ctx.logger.warning(f"Pretty formatting failed, returning raw preview: {e}")
        result_json = json.dumps(result, ensure_ascii=False)
        return f"{name}({args}) → {result_json[:1600]}" + ("…" if len(result_json) > 1600 else "")

# --------------------------------------------------------------------------------------
# Chat protocol handler
# --------------------------------------------------------------------------------------
@chat.on_message(model=ChatMessage, replies=ChatResponse)
async def handle(ctx: Context, sender: str, msg: ChatMessage):
    ctx.logger.info(f"User: {msg.text}")
    try:
        text = _mcp_chat_flow(ctx, msg.text)
        await ctx.send(sender, ChatResponse(text=text))
    except Exception as e:
        await ctx.send(sender, ChatResponse(text=f"Error: {e}"))

# --------------------------------------------------------------------------------------
# Simple REST endpoint to send prompts without uAgents wire format
# POST /chat  { "text": "your question" }  -> { "text": "agent reply" }
# --------------------------------------------------------------------------------------
@agent.on_rest_post("/chat", RestIn, RestOut)
async def rest_chat(ctx: Context, req: RestIn) -> RestOut:
    ctx.logger.info(f"[REST] User: {req.text}")
    try:
        text = _mcp_chat_flow(ctx, req.text)
        return RestOut(text=text)
    except Exception as e:
        return RestOut(text=f"Error: {e}")

# --------------------------------------------------------------------------------------
# Manifest + run
# --------------------------------------------------------------------------------------
agent.include(chat, publish_manifest=True)

if __name__ == "__main__":
    agent.run()
