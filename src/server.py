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


@mcp_server.tool(
    name="read_document_contents",
    description="Read the contents of a document and return it as a string.",
)
def read_document(
    doc_id: str = Field(description="Id of the document to read"),
):
    if doc_id not in docs:
        raise ValueError(f"Document with id {doc_id} not found")

    return docs[doc_id]


@mcp_server.tool(
    name="edit_document",
    description="Edit a document by replacing a string in the documents content with a new string",
)
def edit_document(
    doc_id: str = Field(description="Id of the document that will be edited"),
    old_str: str = Field(
        description="The text to replace. Must match exactly, including whitespace"
    ),
    new_str: str = Field(
        description="The new text to insert in place of the old text"
    ),
):
    if doc_id not in docs:
        raise ValueError(f"Document with id {doc_id} not found")

    docs[doc_id] = docs[doc_id].replace(old_str, new_str)


@mcp_server.tool(
    name="create_document",
    description="Create a new document by providing a document id and the contents of the document",
)
def create_document(
    doc_id: str = Field(description="Id of the document to create"),
    content: str = Field(description="Contents of the document"),
):
    if doc_id in docs:
        raise ValueError(f"Document with id {doc_id} already exists")

    docs[doc_id] = content


@mcp_server.resource("docs://documents", mime_type="application/json")
def list_docs() -> list[str]:
    return list(docs.keys())


@mcp_server.resource("docs://documents/{doc_id}", mime_type="text/plain")
def fetch_doc(doc_id: str) -> str:
    if doc_id not in docs:
        raise ValueError(f"Document with id {doc_id} not found")
    return docs[doc_id]


@mcp_server.prompt(
    name="format",
    description="Rewrites the contents of the document in Markdown format.",
)
def format_document(
    doc_id: str = Field(description="Id of the document to format"),
) -> list[base.Message]:
    prompt = f"""
    Your goal is to reformat a document to be written with markdown syntax.

    The id of the document you need to reformat is:
    <document_id>
    {doc_id}
    </document_id>

    First, fetch the contents of the document with the 'read_document_contents' tool.
    Add in headers, bullet points, tables, etc as necessary. Feel free to add in extra text, but don't change the meaning of the report.
    Use the 'edit_document' tool to edit the document. After the document has been edited, respond with the final version of the doc. Don't explain your changes.
    """

    return [base.UserMessage(prompt)]


@mcp_server.prompt(
    name="summarise",
    description="Summarises the contents of a document.",
)
def summarise_document(
    doc_id: str = Field(description="Id of the document to summarize"),
    summary_type: str = Field(description="Type of the summary to generate")
) -> list[base.Message]:
    prompt = f"""
    Your goal is to summarize the contents of a document.

    The id of the document you need to summarize is:
    <document_id>
    {doc_id}
    </document_id>

    First, fetch the contents of the document with the 'read_document_contents' tool.
    Then, create a {summary_type} summary of the document's contents in your own words.
    """

    return [base.UserMessage(prompt)]


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
