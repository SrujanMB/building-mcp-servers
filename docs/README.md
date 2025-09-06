# Workshop Steps

The MCP server we will complete is taken from the Anthropic [Introduction to Model Context Protocol](https://anthropic.skilljar.com/introduction-to-model-context-protocol) course. Here we will only work on the MCP server, while the course also covers writing an MCP client. The course is free, and I highly recommend taking the time to work through it. For the tool section of this workshop, we will use Microsoft's [AI Toolkit extension](https://marketplace.visualstudio.com/items?itemName=ms-windows-ai-studio.windows-ai-studio) for VS Code as our client. For the rest, there is a custom MCP client based off the one in the Anthropic course, but adapted for using with Github Models.

We will build a document mangement MCP server. In this case we are just running the document store in memory — in the real world this server would be a front for APIs with permanent storage. Interfacing with external APIs is an exercise left to the user. The main file for our MCP server is [`src/server.py`](src/server.py).

Click the links below to be taken to each part of the workshop:

1. [Tools](1.tools.md)
2. [Resources](2.resources.md)
3. [Prompts](3.prompts.md)
