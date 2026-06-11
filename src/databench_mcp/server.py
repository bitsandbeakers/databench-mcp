"""FastMCP server entry point.

Tools are thin orchestration wrappers; all logic lives in databench_mcp.core.
The tool count is asserted at startup so a refactor can never silently drop a tool.
"""
from __future__ import annotations

import asyncio
from typing import Any

from fastmcp import FastMCP

from databench_mcp import __version__
from databench_mcp.tools.eda import eda_summary, sql_query
from databench_mcp.tools.hypothesis import hypothesis_add, hypothesis_list, hypothesis_update
from databench_mcp.tools.ingest import ingest_file, ingest_url
from databench_mcp.tools.profile import profile_table
from databench_mcp.tools.project import project_create, project_list, project_status

mcp = FastMCP("databench")

# Bump this in the same commit that adds or removes a tool.
EXPECTED_TOOL_COUNT = 12


async def ping() -> dict[str, Any]:
    """Liveness check — returns server name and version."""
    return {"status": "ok", "server": "databench-mcp", "version": __version__}


mcp.tool(ping)
mcp.tool(project_create)
mcp.tool(project_list)
mcp.tool(project_status)
mcp.tool(ingest_file)
mcp.tool(ingest_url)
mcp.tool(profile_table)
mcp.tool(sql_query)
mcp.tool(eda_summary)
mcp.tool(hypothesis_add)
mcp.tool(hypothesis_list)
mcp.tool(hypothesis_update)


def _assert_tool_count() -> None:
    tools = asyncio.run(mcp.list_tools())
    names = sorted(t.name for t in tools)
    if len(names) != EXPECTED_TOOL_COUNT:
        raise SystemExit(
            f"databench-mcp tool count drift: expected {EXPECTED_TOOL_COUNT}, "
            f"found {len(names)} ({names})"
        )


def main() -> None:
    _assert_tool_count()
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
