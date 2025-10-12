# mcp.py
from uuid import uuid4
from typing import Any, Dict, List, Optional, Tuple
import json
import requests

MCP_URL = "https://mcp.blockscout.com/mcp"
MCP_HEALTH = "https://mcp.blockscout.com/health"

def mcp_health() -> Tuple[bool, str, int]:
    try:
        r = requests.get(MCP_HEALTH, timeout=6)
        ok = 200 <= r.status_code < 300
        return ok, r.text, r.status_code
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", 0

def _sse_or_json_rpc(method: str, params: Optional[Dict[str, Any]] = None, timeout: int = 25) -> Tuple[bool, Any]:
    """Low-level JSON-RPC call that handles JSON and SSE responses."""
    payload = {"jsonrpc": "2.0", "id": str(uuid4()), "method": method}
    if params is not None:
        payload["params"] = params

    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        "Connection": "keep-alive",
    }

    with requests.post(MCP_URL, headers=headers, json=payload, stream=True, timeout=timeout) as r:
        ctype = (r.headers.get("Content-Type") or "").lower()

        # Plain JSON body
        if "application/json" in ctype and "event-stream" not in ctype:
            try:
                data = r.json()
            except Exception as e:
                return False, f"JSON decode failed: {e}"
            if isinstance(data, dict) and data.get("error") is not None:
                return False, data["error"]
            return True, data.get("result")

        # SSE stream (Server-Sent Events)
        if "text/event-stream" in ctype:
            buffer: List[str] = []
            for raw_line in r.iter_lines(decode_unicode=True):
                line = (raw_line or "").rstrip("\r\n")

                if line == "":  # event boundary
                    if buffer:
                        joined = "\n".join(buffer).strip()
                        buffer.clear()
                        try:
                            obj = json.loads(joined)
                            if isinstance(obj, dict):
                                if obj.get("error") is not None:
                                    return False, obj["error"]
                                if "result" in obj:
                                    return True, obj["result"]
                                # Some servers send a content envelope directly
                                if "content" in obj and obj.get("isError") is False:
                                    return True, obj
                        except Exception:
                            pass
                    continue

                if line.startswith("data:"):
                    buffer.append(line[len("data:"):].strip())

            # Stream ended: flush any remainder
            if buffer:
                try:
                    obj = json.loads("\n".join(buffer))
                    if isinstance(obj, dict):
                        if obj.get("error") is not None:
                            return False, obj["error"]
                        if "result" in obj:
                            return True, obj["result"]
                        if "content" in obj and obj.get("isError") is False:
                            return True, obj
                except Exception:
                    pass
            return False, {"status": r.status_code, "body": "SSE stream ended without result"}

        return False, {"status": r.status_code, "body": r.text[:200]}

def tools_list() -> Tuple[bool, Any]:
    return _sse_or_json_rpc("tools/list")

def tools_call(name: str, arguments: Dict[str, Any]) -> Tuple[bool, Any]:
    ok, res = _sse_or_json_rpc("tools/call", {"name": name, "arguments": arguments})
    # If the server returned a content-envelope as the top-level
    if not ok and isinstance(res, dict) and res.get("isError") is False and "content" in res:
        return True, res
    return ok, res

def unwrap_content(enveloped: Any) -> Any:
    """
    For payloads like {'content':[{'text':'{...json...}'}], 'isError': False},
    parse and return the inner JSON; otherwise return unchanged.
    """
    if isinstance(enveloped, dict) and isinstance(enveloped.get("content"), list):
        texts = [c.get("text", "") for c in enveloped["content"] if isinstance(c, dict)]
        blob = "\n".join(texts).strip()
        if blob:
            try:
                return json.loads(blob)
            except Exception:
                return {"raw_text": blob}
    return enveloped