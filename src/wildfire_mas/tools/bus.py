"""ToolBus: how agents call tools.

  InProcessBus  - calls Toolkit methods directly (fast; used for tests and model sweeps)
  MCPBus        - talks to the four MCP servers (stdio subprocesses or HTTP URLs)

Both record every call (tool, args, latency, ok) for the answer trace.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass
from typing import Any

from .toolkit import TOOL_METHODS, Toolkit

SERVER_OF = {"firms": "firms", "weather": "weather", "kg": "graphdb", "rag": "rag", "vision": "rag"}
MODULE_OF = {
    "firms": "wildfire_mas.mcp_servers.firms_server",
    "weather": "wildfire_mas.mcp_servers.weather_server",
    "graphdb": "wildfire_mas.mcp_servers.graphdb_server",
    "rag": "wildfire_mas.mcp_servers.rag_server",
}


@dataclass
class ToolCall:
    tool: str
    args: dict
    latency_s: float
    ok: bool
    error: str | None = None


class ToolError(RuntimeError):
    pass


class BaseBus:
    def __init__(self):
        self.calls: list[ToolCall] = []

    async def call(self, tool: str, args: dict | None = None) -> dict:
        args = args or {}
        t0 = time.perf_counter()
        try:
            out = await self._call(tool, args)
            self.calls.append(ToolCall(tool, _redact(args), time.perf_counter() - t0, True))
            return out
        except Exception as e:  # noqa: BLE001
            self.calls.append(ToolCall(tool, _redact(args), time.perf_counter() - t0, False, str(e)))
            raise ToolError(f"{tool} failed: {e}") from e

    async def _call(self, tool: str, args: dict) -> dict:  # pragma: no cover
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


def _redact(args: dict) -> dict:
    return {k: ("<image>" if k == "image_b64" else v) for k, v in args.items()}


class InProcessBus(BaseBus):
    def __init__(self, toolkit: Toolkit):
        super().__init__()
        self.tk = toolkit

    async def _call(self, tool: str, args: dict) -> dict:
        if tool not in TOOL_METHODS:
            raise ToolError(f"unknown tool {tool}")
        fn = getattr(self.tk, TOOL_METHODS[tool])
        return await asyncio.to_thread(fn, **args)


class MCPBus(BaseBus):
    """servers: {"firms": "http://host:port/mcp" | MCPServer instance | None}.
    None = launch the server module over stdio; an MCPServer instance connects in-process."""

    def __init__(self, servers: dict[str, Any] | None = None):
        super().__init__()
        self.servers = servers or {k: None for k in MODULE_OF}
        self._clients: dict[str, Any] = {}
        self._owners: list[asyncio.Task] = []
        self._lock: asyncio.Lock | None = None
        self._closing: asyncio.Event | None = None

    def _target(self, server: str):
        from mcp import StdioServerParameters

        target = self.servers.get(server)
        if target is None:
            import os

            from ..config import project_root

            # pass the environment through (MAP_KEY, GraphDB, model URLs) and pin the project root
            env = {**os.environ, "WFMAS_HOME": str(project_root())}
            target = StdioServerParameters(command=sys.executable, args=["-m", MODULE_OF[server]], env=env,
                                           cwd=str(project_root()))
        return target

    async def _client(self, server: str):
        """Each MCP connection is opened AND closed by one dedicated owner task.

        The MCP client uses anyio task groups, which must be exited in the task that
        entered them. Agents call tools from many parallel tasks, so connections are not
        opened in the caller's task; the owner task keeps the connection alive until
        aclose() and the callers just use the session.
        """
        if server in self._clients:
            return self._clients[server]
        if self._lock is None:
            self._lock, self._closing = asyncio.Lock(), asyncio.Event()
        async with self._lock:
            if server in self._clients:
                return self._clients[server]
            from mcp import Client

            ready = asyncio.Event()
            box: dict[str, Any] = {}
            closing = self._closing

            async def owner():
                try:
                    async with Client(self._target(server)) as client:
                        box["client"] = client
                        ready.set()
                        await closing.wait()
                except BaseException as e:  # noqa: BLE001 - surfaced to the first caller
                    box.setdefault("error", e)
                    ready.set()

            self._owners.append(asyncio.create_task(owner(), name=f"mcp-{server}"))
            await ready.wait()
            if "client" not in box:
                raise ToolError(f"could not start MCP server '{server}': {box.get('error')!r}")
            self._clients[server] = box["client"]
            return box["client"]

    async def _call(self, tool: str, args: dict) -> dict:
        prefix, name = tool.split(".", 1)
        client = await self._client(SERVER_OF[prefix])
        res = await client.call_tool(name, args)
        if getattr(res, "is_error", False) or getattr(res, "isError", False):
            raise ToolError(_text(res))
        structured = getattr(res, "structured_content", None) or getattr(res, "structuredContent", None)
        if isinstance(structured, dict):
            return structured.get("result", structured) if set(structured) == {"result"} else structured
        return json.loads(_text(res))

    async def aclose(self) -> None:
        if self._closing is not None:
            self._closing.set()
        if self._owners:
            await asyncio.gather(*self._owners, return_exceptions=True)
        self._owners.clear()
        self._clients.clear()


def _text(res) -> str:
    return "".join(getattr(c, "text", "") for c in getattr(res, "content", []) or [])
