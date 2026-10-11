"""MCP servers (mcp>=2, MCPServer API). Run one with:
    python -m wildfire_mas.mcp_servers.firms_server              # stdio
    python -m wildfire_mas.mcp_servers.firms_server --http 8101  # streamable HTTP
"""
from __future__ import annotations

import argparse

import anyio


def run(server) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", type=int, default=None, help="serve streamable HTTP on this port")
    a = ap.parse_args()
    if a.http:
        server.run("streamable-http", port=a.http)
    else:
        server.run("stdio")


async def in_thread(fn, *args, **kwargs):
    return await anyio.to_thread.run_sync(lambda: fn(*args, **kwargs))
