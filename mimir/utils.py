# utils.py
import re
import json
from math import log10
from typing import Any, Dict

HEX_ADDR_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")

def is_hex_address(s: str) -> bool:
    return bool(HEX_ADDR_RE.match(s or ""))

def trim_addr(a: str) -> str:
    return a[:6] + "…" + a[-4:] if isinstance(a, str) and a.startswith("0x") and len(a) > 12 else a

def fmt_eth(wei_like: str | int | None) -> str:
    if wei_like is None:
        return "0 ETH"
    try:
        w = int(wei_like)
    except Exception:
        return str(wei_like)
    eth = w / 10**18
    if eth == 0:
        return "0 ETH"
    digits = max(0, 6 - int(max(1, log10(abs(eth)))))
    return f"{eth:.{digits}f} ETH"

def maybe_usd(wei_like: str | int | None, price_str: str | None) -> str:
    if wei_like is None or not price_str:
        return ""
    try:
        w = int(wei_like)
        px = float(price_str)
        usd = (w / 10**18) * px
        return f" (~${usd:,.2f})"
    except Exception:
        return ""

def extract_json_from_content_like(obj: Any) -> Any:
    """
    If server wraps JSON inside string fields (e.g., content->text),
    try parsing the string to JSON; otherwise return obj unchanged.
    """
    if isinstance(obj, dict) and isinstance(obj.get("content"), list):
        try:
            texts = [c.get("text", "") for c in obj["content"] if isinstance(c, dict)]
            blob = "\n".join(texts).strip()
            return json.loads(blob)
        except Exception:
            return obj
    return obj

def pretty(obj: Any) -> str:
    # nicer text for common MCP envelopes
    if isinstance(obj, dict) and isinstance(obj.get("content"), list):
        parts = [str(c.get("text", "")) for c in obj["content"] if isinstance(c, dict)]
        msg = " ".join(parts).strip()
        if msg:
            try:
                return json.dumps(json.loads(msg), ensure_ascii=False, indent=2)
            except Exception:
                return msg
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)