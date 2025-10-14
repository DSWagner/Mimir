
#!/usr/bin/env python3
"""
Simple CLI client for the Mimir MCP Agent.

Usage:
    python client.py "What is the balance on address 0x... on Ethereum?"

You can also override the agent base URL with:
    AGENT_BASE_URL=http://127.0.0.1:8000 python client.py "..."

Or via flag:
    python client.py --base-url http://127.0.0.1:8000 "..."

This sends a POST /chat { "text": "<your prompt>" } and prints the agent's reply.
"""
import os
import sys
import json
import argparse
from typing import Any, Dict
from dotenv import load_dotenv
import requests

DEFAULT_BASE = "http://127.0.0.1:8000"

def post_chat(base_url: str, prompt: str, timeout: int = 60) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/chat"
    resp = requests.post(url, json={"text": prompt}, timeout=timeout)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        # surface server error body if present
        body = resp.text
        raise SystemExit(f"HTTP {resp.status_code} while calling {url}:\n{body}") from e
    try:
        return resp.json()
    except Exception:
        # Fallback: return text as a JSON-compat object
        return {"text": resp.text}

def main() -> None:
    load_dotenv()  # allow AGENT_BASE_URL from .env
    parser = argparse.ArgumentParser(description="Send a prompt to the local Mimir MCP Agent (/chat).")
    parser.add_argument("prompt", help="Your prompt text. Wrap in quotes.")
    parser.add_argument("--base-url", default=os.getenv("AGENT_BASE_URL", DEFAULT_BASE),
                        help=f"Agent base URL (default: %(default)s). "
                             f'You can also set AGENT_BASE_URL env var.')
    parser.add_argument("--timeout", type=int, default=int(os.getenv("CLIENT_TIMEOUT", "60")),
                        help="HTTP timeout in seconds (default: %(default)s)")
    args = parser.parse_args()

    data = post_chat(args.base_url, args.prompt, timeout=args.timeout)
    # Expecting {"text": "..."} as per agent REST contract
    text = data.get("text")
    if isinstance(text, str):
        print(text)
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
