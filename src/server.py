# Copyright (c)  Jerome Brown, Anthropic
# This file is part of the project licensed under the MIT License. See the
# project root `LICENSE` file for the full text.

import os
import sys
import signal
from pydantic import Field
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base

mcp_server = FastMCP("DocumentMCP")


docs = {
    "deposition.md": "This deposition covers the testimony of Angela Smith, P.E.",
    "report.pdf": "The report details the state of a 20m condenser tower.",
    "financials.docx": "These financials outline the project's budget and expenditures.",
    "outlook.pdf": "This document presents the projected future performance of the system.",
    "plan.md": "The plan outlines the steps for the project's implementation.",
    "spec.txt": "These specifications define the technical requirements for the equipment.",
}

# TODO: Write a tool to read a doc
# TODO: Write a tool to edit a doc
# TODO: Write a resource to return all doc id's
# TODO: Write a resource to return the contents of a particular doc
# TODO: Write a prompt to rewrite a doc in markdown format
# TODO: Write a prompt to summarize a doc


def signal_handler(sig, frame):
    """Handle Ctrl-C gracefully"""
    print("\n🛑 Received interrupt signal. Shutting down MCP server gracefully...")
    sys.exit(0)


if __name__ == "__main__":
    """Main entry point"""
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    transport_type = sys.argv[1] if len(sys.argv) > 1 else None
    mcp_server.settings.log_level = os.environ.get("LOG_LEVEL", "DEBUG")
    if transport_type == "sse":
        mcp_server.settings.port = int(os.environ.get("PORT", 3001))
        mcp_server.run(transport="sse")
    elif transport_type == "http":
        mcp_server.settings.port = int(os.environ.get("PORT", 8000))
        mcp_server.run(transport="streamable-http")
    elif transport_type == "stdio":
        mcp_server.run(transport="stdio")
    else:
        print("Invalid transport type. Use 'http', 'sse', or 'stdio'.")
        sys.exit(1)
