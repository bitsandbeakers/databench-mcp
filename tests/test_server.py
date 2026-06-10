"""Server skeleton tests."""

from databench_mcp import __version__
from databench_mcp.server import EXPECTED_TOOL_COUNT, mcp, ping


async def test_ping_returns_ok():
    result = await ping()
    assert result["status"] == "ok"
    assert result["server"] == "databench-mcp"
    assert result["version"] == __version__


async def test_tool_count_matches_expected():
    tools = await mcp.list_tools()
    names = sorted(t.name for t in tools)
    assert len(names) == EXPECTED_TOOL_COUNT, names
