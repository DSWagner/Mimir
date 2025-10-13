import os
import requests
from typing import Dict, Any, List, Optional, Union

# IMPORTANT: default points to your Dockerized MCP server mapped to 8001
MCP_BASE_URL = os.getenv("MCP_BASE_URL", "http://localhost:8001").rstrip("/")
TIMEOUT = int(os.getenv("MCP_TIMEOUT", "30"))

def _extract_tools(payload: Union[Dict[str, Any], List[Any]]) -> List[Dict[str, Any]]:
    """
    Accepts either:
      - {"tools": [...]} OR
      - [...]
    Returns the tools list or raises a ValueError if the shape is unexpected.
    """
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        tools = payload.get("tools")
        if isinstance(tools, list):
            return tools
        # Some servers might return under a different key; fall back if dict itself is a list-like
        if isinstance(payload, list):
            return payload  # defensive (shouldn't happen)
    raise ValueError(f"Unexpected tools payload: {payload!r}")

class MCPClient:
    def __init__(self, base_url: Optional[str] = None, timeout: int = TIMEOUT):
        self.base = (base_url or MCP_BASE_URL).rstrip("/")
        self.timeout = timeout

    def _get(self, path: str) -> requests.Response:
        url = f"{self.base}{path}"
        r = requests.get(url, timeout=self.timeout)
        if r.status_code >= 400:
            # Raise with body included in the exception chain
            try:
                _ = r.json()
            except Exception:
                pass
            r.raise_for_status()
        return r

    def _post(self, path: str, json_payload: Dict[str, Any]) -> requests.Response:
        url = f"{self.base}{path}"
        r = requests.post(url, json=json_payload, timeout=self.timeout)
        if r.status_code >= 400:
            try:
                _ = r.json()
            except Exception:
                pass
            r.raise_for_status()
        return r

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        Prefer Blockscout MCP REST paths:
          - NEW: GET /v1/tools
          - Fallback: GET /tools/list
        Returns a list of tool objects.
        """
        # Try new path
        try:
            r = self._get("/v1/tools")
            data = r.json()
            return _extract_tools(data)
        except requests.HTTPError as e:
            if e.response is None or e.response.status_code != 404:
                raise
        except ValueError:
            # Payload shape unexpected; try legacy path
            pass

        # Legacy path
        r = self._get("/tools/list")
        data = r.json()
        return _extract_tools(data)

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prefer per-tool GET endpoints:
        - GET /v1/unlock_blockchain_analysis                 (special case)
        - GET /v1/{tool_name}?<query params>
        Fallbacks (for other MCP servers/builds):
        - POST /v1/tools/call {name, arguments}
        - POST /tools/call    {name, arguments} (legacy)
        """
        # --- Preferred: GET per-tool endpoint ---
        # Special path for unlock: server exposes it without the double underscores
        if name == "__unlock_blockchain_analysis__":
            get_path = "/v1/unlock_blockchain_analysis"
        else:
            get_path = f"/v1/{name}"

        # Try GET with query params
        try:
            url = f"{self.base}{get_path}"
            r = requests.get(url, params=arguments, timeout=self.timeout)
            if r.status_code == 200:
                return r.json()
            # If not found, try fallbacks below
            if r.status_code not in (404, 405):
                r.raise_for_status()
        except requests.HTTPError as e:
            # Only fall back on 404/405; surface other errors
            if e.response is None or e.response.status_code not in (404, 405):
                raise

        # --- Fallback A: POST hub endpoint /v1/tools/call ---
        try:
            r = self._post("/v1/tools/call", {"name": name, "arguments": arguments})
            return r.json()
        except requests.HTTPError as e:
            if e.response is None or e.response.status_code != 404:
                raise

        # --- Fallback B: legacy /tools/call ---
        r = self._post("/tools/call", {"name": name, "arguments": arguments})
        return r.json()

