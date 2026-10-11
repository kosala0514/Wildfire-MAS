"""rag-mcp: document passages, tables and figures, plus the vision model for figures/scenes."""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from ..tools.toolkit import Toolkit
from . import in_thread, run

server = MCPServer("rag-mcp", instructions="Multimodal wildfire document index and figure reader.")
_tk = Toolkit()


@server.tool()
async def search(query: str, k: int | None = None, kinds: list[str] | None = None) -> dict:
    """Top passages; kinds filters to text / table / figure."""
    return await in_thread(_tk.rag_search, query=query, k=k, kinds=kinds)


@server.tool()
async def read_figure(image_path: str, question: str) -> dict:
    """Ask the vision model to read a specific value or trend from a figure."""
    return await in_thread(_tk.vision_read_figure, image_path=image_path, question=question)


@server.tool()
async def scene_graph(image_path: str, concepts: list[str]) -> dict:
    """Describe an image using only the given vocabulary concepts."""
    return await in_thread(_tk.vision_scene_graph, image_path=image_path, concepts=concepts)


if __name__ == "__main__":
    run(server)
