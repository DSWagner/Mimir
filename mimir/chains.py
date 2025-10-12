# chains.py
from typing import Any, Dict, List, Tuple, Optional
from mcp import tools_call, unwrap_content

# cached chains list
_CHAINS: Optional[List[Dict[str, Any]]] = None

def get_chains() -> Tuple[bool, List[Dict[str, Any]] | str]:
    global _CHAINS
    if _CHAINS is not None:
        return True, _CHAINS
    ok, res = tools_call("get_chains_list", {})
    if not ok:
        return False, str(res)
    data = unwrap_content(res)
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        _CHAINS = data["data"]
        return True, _CHAINS
    if isinstance(data, list):
        _CHAINS = data
        return True, _CHAINS
    return False, f"Unexpected chains payload: {data}"

def _text_match(hay: Any, needle: str) -> bool:
    n = needle.lower()
    if isinstance(hay, dict):
        return any(_text_match(v, needle) for v in hay.values())
    if isinstance(hay, list):
        return any(_text_match(v, needle) for v in hay)
    if isinstance(hay, str):
        return n in hay.lower()
    return False

_DEFAULT_CHAIN_PREF = ("ethereum", "mainnet", "eth")

def select_chain_id(user_token: Optional[str]) -> Tuple[bool, str | str]:
    """
    Resolve a user-provided chain token (name or numeric) to chain_id **string**
    using get_chains_list. If token is None, pick a sensible default.
    """
    ok, chains = get_chains()
    if not ok:
        return False, chains  # error str
    chains_list = chains  # type: ignore

    # Numeric id (string) path
    if user_token and user_token.isdigit():
        for ch in chains_list:
            if str(ch.get("chain_id")) == user_token:
                return True, user_token
        return False, f"Unknown chain_id: {user_token}"

    # No token: pick default by preference keywords
    if not user_token:
        for pref in _DEFAULT_CHAIN_PREF:
            for ch in chains_list:
                if _text_match(ch, pref):
                    return True, str(ch.get("chain_id"))
        if chains_list:
            return True, str(chains_list[0].get("chain_id"))
        return False, "No chains available from MCP."

    # Name-ish token
    token = user_token.strip().lower()
    # exact matches first
    for ch in chains_list:
        for key in ("name", "slug", "chain", "network", "shortName", "title"):
            v = ch.get(key)
            if isinstance(v, str) and v.lower() == token:
                return True, str(ch.get("chain_id"))
    # contains-match fallback
    for ch in chains_list:
        if _text_match(ch, token):
            return True, str(ch.get("chain_id"))
    return False, f"Could not map '{user_token}' to any chain. Try @mimir chains"

def chain_name_by_id(chain_id_str: str) -> str:
    ok, chains = get_chains()
    if not ok:
        return f"chain_id {chain_id_str}"
    for ch in chains:  # type: ignore
        if str(ch.get("chain_id")) == str(chain_id_str):
            return ch.get("name") or ch.get("title") or ch.get("chain") or f"chain_id {chain_id_str}"
    return f"chain_id {chain_id_str}"
