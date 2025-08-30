# VS Code Dev Days — Building MCP Servers

This repository contains the base code for the workshop initially run as part of the VS Code Dev Days 2025 in Auckland, NZ. 
The MCP Server code is taken from Anthropic's [Introduction to Model Context Protocol](https://anthropic.skilljar.com/introduction-to-model-context-protocol) course.
Parts of the workshop, including the base configuration, are drawn from the [_Building an MCP Server with AI Toolkit_](https://github.com/microsoft/mcp-for-beginners/tree/main/10-StreamliningAIWorkflowsBuildingAnMCPServerWithAIToolkit) section of Microsoft's [MCP for Beginners](https://github.com/microsoft/mcp-for-beginners) curriculum.

To run the client for testing the server you will need a GitHub PAT token with the `models:read` permisson.
[Generate a new fine-grained PAT](https://github.com/settings/personal-access-tokens/new). Then either create an environment variable called `GITHUB_TOKEN` with you token as the value, or create a `.env` file in the root of the repository and set `GITHUB_TOKEN` there.

## Prerequisites

This workshop is based around [Visual Studio Code](https://code.visualstudio.com).
To make things easier, all the pre-requisites are configured within a [Dev Container](https://containers.dev), which could be used either locally or in [GitHub Codespaces](https://docs.github.com/en/codespaces/setting-up-your-project-for-codespaces/adding-a-dev-container-configuration/introduction-to-dev-containers). To use the Dev Container locally, you need a docker-compatible runtime and the [Dev Container extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

If you wish to run the code locally, you will require:
- Python 3.10+
- Node.js and npm

Once you have these installed, the required packages can be installed by running the following commands:

```shell
# Install Python pre-requisites. Will create a virtual environment if one doesn't already exist.
uv sync --all-extras

# Install NodeJS pre-requisites for the inspector
cd inspector
npm install
``` 

Then activate the Python Virtual Environment (VS Code should prompt you).
