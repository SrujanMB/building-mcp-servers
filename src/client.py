# Copyright (c) Jerome Brown
# This file is part of the project licensed under the MIT License. See the
# project root `LICENSE` file for the full text.

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from typing import Any, Optional, Sequence

# prompt_toolkit for async interactive autocompletion
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.patch_stdout import patch_stdout
import re

from dotenv import load_dotenv

from mcp import ClientSession, types
from mcp.client.streamable_http import streamablehttp_client
from pydantic import AnyUrl

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage, ToolMessage
from azure.core.credentials import AzureKeyCredential

load_dotenv()

# Configure GitHub Models
GITHUB_MODELS_ENDPOINT = "https://models.github.ai/inference"
GITHUB_MODEL = os.environ.get("GITHUB_MODEL", "openai/gpt-4o-mini")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

# MCP Server configuration
MCP_SERVER_URL = "http://localhost:8000/mcp"


class MCPClient:
    """Helper to manage an MCP HTTP client session."""

    def __init__(self, server_url: str = MCP_SERVER_URL):
        self._server_url = server_url
        self._session: Optional[ClientSession] = None
        self._exit_stack: AsyncExitStack = AsyncExitStack()

    async def connect(self) -> None:
        """Connect to the MCP server via HTTP."""
        transport = await self._exit_stack.enter_async_context(
            streamablehttp_client(url=self._server_url)
        )
        # The third element is a callable for session ID
        read_stream, write_stream, _ = transport
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()

    def session(self) -> ClientSession:
        if self._session is None:
            raise ConnectionError(
                "Client session not initialized. Call connect first.")
        return self._session

    async def list_tools(self) -> list[types.Tool]:
        result = await self.session().list_tools()
        return result.tools

    async def call_tool(self, tool_name: str, tool_input) -> types.CallToolResult | None:
        return await self.session().call_tool(tool_name, tool_input)

    async def list_prompts(self) -> list[types.Prompt]:
        result = await self.session().list_prompts()
        return result.prompts

    async def get_prompt(self, prompt_name: str, args: dict[str, str]):
        result = await self.session().get_prompt(prompt_name, args)
        return result.messages

    async def read_resource(self, uri: str) -> Any:
        result = await self.session().read_resource(AnyUrl(uri))
        resource = result.contents[0]
        if isinstance(resource, types.TextResourceContents):
            if resource.mimeType == "application/json":
                return json.loads(resource.text)
            return resource.text

    async def list_resources(self) -> list[types.Resource]:
        result = await self.session().list_resources()
        return result.resources

    async def cleanup(self) -> None:
        await self._exit_stack.aclose()
        self._session = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()


class MCPCompleter(Completer):
    """A small prompt_toolkit completer that fetches prompts and resources from the MCP server.

    Usage: create an instance with the connected `MCPClient`, call `await completer.refresh_once()`
    before starting the prompt loop, and optionally run `asyncio.create_task(completer.periodic_refresh())`
    to keep the cache fresh.
    """

    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client
        self._prompts: list[str] = []
        self._resources: list[str] = []
        # document/resource instances (e.g., files inside docs://documents)
        self._resource_instances: list[str] = []
        # background expansion task handle (if running)
        self._expansion_task: Optional[asyncio.Task] = None

    async def refresh_once(self):
        try:
            prompts = await self.mcp_client.list_prompts()
            resources = await self.mcp_client.list_resources()
            self._prompts = [
                p.name for p in prompts if getattr(p, "name", None)]
            # Cast resource URIs to strings (AnyUrl -> str)
            self._resources = [str(getattr(r, "uri"))
                               for r in resources if getattr(r, "uri", None)]
            # Reset any previously-expanded instances so expansion will be lazy
            self._resource_instances = []
        except Exception:
            # keep existing lists on error
            pass

    async def periodic_refresh(self, interval: int = 60):
        while True:
            await asyncio.sleep(interval)
            await self.refresh_once()

    def get_completions(self, document, complete_event):
        try:
            text = document.text_before_cursor or ""
            # capture optional trigger (/ or #) and the token after it
            m = re.search(r'([/#]?)([^\s/#]*)$', text)
            if not m:
                return
            trigger, token = m.group(1), m.group(2)
            token_l = token.lower()

            # start_position: replace only the token (and trigger if present)
            start_pos = -len((trigger or "") + token)

            # choose candidates by trigger: '#' -> resource instances, '/' -> prompts
            if trigger == '#':
                # if instances not yet expanded, schedule a background expansion and return no completions for now
                if not self._resource_instances:
                    if not self._expansion_task or self._expansion_task.done():
                        try:
                            # schedule expansion but do not await here
                            self._expansion_task = asyncio.create_task(
                                self._ensure_instances())
                        except Exception:
                            # event loop may not allow create_task here; ignore
                            pass
                    return
                candidates = list(self._resource_instances)
            elif trigger == '/':
                candidates = [str(x) for x in self._prompts]
            else:
                candidates = list(self._resource_instances) + \
                    [str(x) for x in self._prompts] + list(self._resources)
            for cand in candidates:
                if not cand:
                    continue

                # If there's no token, optionally show suggestions (here we show all)
                if not token_l:
                    display_text = (trigger + cand) if trigger else cand
                    yield Completion(display_text, start_position=start_pos, display_meta="resource")
                    continue

                if token_l in cand.lower():
                    display_text = (trigger + cand) if trigger else cand
                    yield Completion(display_text, start_position=start_pos, display_meta=("prompt" if trigger == '/' else "resource"))
        except Exception:
            # keep prompt_toolkit stable on errors
            return

    async def _ensure_instances(self):
        """Populate `self._resource_instances` by reading docs:// resources lazily."""
        try:
            instances: list[str] = []
            # ensure we have a current resource list
            if not self._resources and self.mcp_client:
                try:
                    await self.refresh_once()
                except Exception:
                    pass

            for uri in self._resources:
                if not str(uri).startswith("docs://"):
                    continue
                try:
                    contents = await self.mcp_client.read_resource(str(uri))
                    if isinstance(contents, list):
                        for item in contents:
                            if isinstance(item, dict):
                                name = item.get("name") or item.get(
                                    "filename") or item.get("id")
                                if name:
                                    instances.append(str(name))
                                else:
                                    instances.append(str(item))
                            else:
                                instances.append(str(item))
                    elif isinstance(contents, dict):
                        for k in ("documents", "items", "files"):
                            if k in contents and isinstance(contents[k], list):
                                for item in contents[k]:
                                    if isinstance(item, dict):
                                        name = item.get("name") or item.get(
                                            "filename") or item.get("id")
                                        if name:
                                            instances.append(str(name))
                                        else:
                                            instances.append(str(item))
                                    else:
                                        instances.append(str(item))
                                break
                except Exception:
                    # skip problematic resources
                    pass

            # Deduplicate while preserving order
            seen = set()
            deduped: list[str] = []
            for i in instances:
                if i not in seen:
                    seen.add(i)
                    deduped.append(i)
            self._resource_instances = deduped
        finally:
            # allow a new expansion task to be scheduled in the future
            self._expansion_task = None


class InteractiveMCPClient:
    """Interactive CLI client to demonstrate MCP Server capabilities."""

    def __init__(self):
        self.chat_client = None
        self.mcp_client = None
        self.conversation_history = []
        # cached MCP capabilities to avoid network calls on every user input
        self._mcp_cache = {
            "tools": [],
            "prompts": [],
            "resources": [],
        }

    async def initialize(self):
        """Initialize both the chat client and MCP client."""
        try:
            self.chat_client = ChatClient()
            print("✓ GitHub Models chat client initialized")
        except RuntimeError as e:
            print(f"✗ Failed to initialize chat client: {e}")
            return False

        try:
            self.mcp_client = MCPClient()
            await self.mcp_client.connect()
            print("✓ MCP client connected to server")
        except Exception as e:
            print(f"✗ Failed to connect to MCP server: {e}")
            return False

        # populate an initial cache of capabilities
        try:
            await self.refresh_mcp_cache()
        except Exception:
            # non-fatal; continue without cache
            pass

        return True

    async def show_mcp_capabilities(self):
        """Display available tools, prompts, and resources."""
        print("\n🔧 Available MCP Tools:")
        try:
            tools = await self.mcp_client.list_tools()
            for tool in tools:
                print(f"  • {tool.name}: {tool.description}")
        except Exception as e:
            print(f"  Error listing tools: {e}")

        print("\n📝 Available MCP Prompts:")
        try:
            prompts = await self.mcp_client.list_prompts()
            for prompt in prompts:
                print(f"  • {prompt.name}: {prompt.description}")
        except Exception as e:
            print(f"  Error listing prompts: {e}")

        print("\n📚 Available MCP Resources:")
        try:
            resources = await self.mcp_client.list_resources()
            for resource in resources:
                print(
                    f"  • {resource.uri}: {resource.name or 'No description'}")
        except Exception as e:
            print(f"  Error listing resources: {e}")

    async def refresh_mcp_cache(self):
        """Refresh cached tools, prompts and resources from the MCP server."""
        tools = await self.mcp_client.list_tools()
        prompts = await self.mcp_client.list_prompts()
        resources = await self.mcp_client.list_resources()

        self._mcp_cache["tools"] = tools
        self._mcp_cache["prompts"] = prompts
        self._mcp_cache["resources"] = resources

    async def _periodic_cache_refresh(self, interval: int = 60):
        while True:
            try:
                await asyncio.sleep(interval)
                await self.refresh_mcp_cache()
            except asyncio.CancelledError:
                break
            except Exception:
                # swallow errors to avoid crashing background task
                await asyncio.sleep(interval)

    async def handle_user_input(self, user_input: str) -> str:
        """Process user input and potentially call MCP tools."""

        # Check if user input is a prompt request (starts with /)
        if user_input.startswith('/'):
            return await self._handle_prompt_request(user_input)

        # Add user message to conversation
        self.conversation_history.append(
            {"role": "user", "content": user_input})

        # Create system message with MCP capabilities using cached values when possible
        tools = self._mcp_cache.get("tools") or []
        if not tools and self.mcp_client:
            try:
                tools = await self.mcp_client.list_tools()
            except Exception:
                tools = []

        resources = self._mcp_cache.get("resources") or []
        if not resources and self.mcp_client:
            try:
                resources = await self.mcp_client.list_resources()
            except Exception:
                resources = []

        # Get prompts for system message
        prompts = self._mcp_cache.get("prompts") or []
        if not prompts and self.mcp_client:
            try:
                prompts = await self.mcp_client.list_prompts()
            except Exception:
                prompts = []

        # Convert MCP tools to Azure AI tool format
        azure_tools = []
        for tool in tools:
            tool_def = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }

            # Convert MCP tool input schema to Azure AI format
            if hasattr(tool, 'inputSchema') and tool.inputSchema:
                if isinstance(tool.inputSchema, dict):
                    if 'properties' in tool.inputSchema:
                        tool_def["function"]["parameters"]["properties"] = tool.inputSchema['properties']
                    if 'required' in tool.inputSchema:
                        tool_def["function"]["parameters"]["required"] = tool.inputSchema['required']
                else:
                    # Handle case where inputSchema is an object with attributes
                    if hasattr(tool.inputSchema, 'properties') and tool.inputSchema.properties:
                        tool_def["function"]["parameters"]["properties"] = tool.inputSchema.properties
                    if hasattr(tool.inputSchema, 'required') and tool.inputSchema.required:
                        tool_def["function"]["parameters"]["required"] = tool.inputSchema.required

            azure_tools.append(tool_def)

        tool_descriptions = "\n".join(
            [f"- {tool.name}: {tool.description}" for tool in tools])

        prompt_descriptions = "\n".join(
            [f"- {prompt.name}: {prompt.description}" for prompt in prompts])

        resource_descriptions = "\n".join(
            [f"- {getattr(resource, 'uri', str(resource))}: {getattr(resource, 'name', 'No description') or 'No description'}" for resource in resources])

        system_message = {
            "role": "system",
            "content": f"""You are a helpful assistant with access to MCP (Model Context Protocol) tools and resources.

Available tools:
{tool_descriptions}

Available prompts (use with /prompt_name format):
{prompt_descriptions}

Available resources:
{resource_descriptions}

When the user asks about documents or needs to perform document operations, use the appropriate tools.
You have access to tools for reading, editing, and creating documents.

If the user asks about listing documents or resources, you can access the docs://documents resource.
If the user asks about reading a specific document, use the read_document_contents tool.
If the user asks to edit a document, use the edit_document tool.

Use the tools when appropriate to help the user with their requests."""
        }

        # Prepare messages for the chat client
        messages = [system_message] + self.conversation_history

        # Get response from the AI model with tools
        response = await self.chat_client.chat_with_tools(messages, azure_tools, self.mcp_client)

        # Add assistant response to conversation
        self.conversation_history.append(
            {"role": "assistant", "content": response})

        return response

    async def _handle_prompt_request(self, user_input: str) -> str:
        """Handle MCP prompt request from user input starting with /."""
        try:
            # Remove the leading '/' and parse the prompt request
            prompt_text = user_input[1:].strip()

            if not prompt_text:
                return "Please specify a prompt name after '/' (e.g., /summarize)"

            # Parse prompt name and potential arguments
            # Format: /prompt_name arg1 arg2 or /prompt_name arg1=value1 arg2=value2
            parts = prompt_text.split()
            if not parts:
                return "Please specify a prompt name after '/'"

            prompt_name = parts[0]

            # Get the prompt definition to understand its parameters
            prompts = self._mcp_cache.get("prompts") or []
            if not prompts:
                prompts = await self.mcp_client.list_prompts()

            target_prompt = None
            for prompt in prompts:
                if prompt.name == prompt_name:
                    target_prompt = prompt
                    break

            if not target_prompt:
                return f"Prompt '{prompt_name}' not found. Available prompts: {[p.name for p in prompts]}"

            # Parse arguments - handle both positional and named
            prompt_args = {}
            positional_args = []
            named_args = {}

            for part in parts[1:]:
                if '=' in part:
                    key, value = part.split('=', 1)
                    named_args[key.strip()] = value.strip()
                else:
                    # strip the leading # if the argument was called as a resource
                    positional_args.append(part.lstrip("#"))

            # Map positional arguments to parameter names based on prompt definition
            if positional_args:
                param_names = []

                # Try to extract parameter names from the prompt's arguments list
                if hasattr(target_prompt, 'arguments') and target_prompt.arguments:
                    if isinstance(target_prompt.arguments, list):
                        # arguments is a list of PromptArgument objects
                        param_names = [
                            arg.name for arg in target_prompt.arguments if hasattr(arg, 'name')]
                    elif isinstance(target_prompt.arguments, dict):
                        if 'properties' in target_prompt.arguments:
                            param_names = list(
                                target_prompt.arguments['properties'].keys())
                        else:
                            param_names = list(target_prompt.arguments.keys())
                    elif hasattr(target_prompt.arguments, 'properties'):
                        param_names = list(
                            target_prompt.arguments.properties.keys())

                # Fallback: use known parameter mappings for specific prompts
                if not param_names:
                    if prompt_name == "format":
                        param_names = ["doc_id"]
                    elif prompt_name == "summarise":
                        param_names = ["doc_id", "summary_type"]
                    else:
                        # Generic fallback: use arg1, arg2, etc.
                        param_names = [
                            f"arg{i+1}" for i in range(len(positional_args))]

                # Map positional arguments to parameter names
                for i, arg in enumerate(positional_args):
                    if i < len(param_names):
                        prompt_args[param_names[i]] = arg
                    else:
                        # If we have more positional args than known parameters, use generic names
                        prompt_args[f"arg{i+1}"] = arg

            # Add named arguments (they override positional if there's a conflict)
            prompt_args.update(named_args)

            print(
                f"📝 Using MCP prompt: {prompt_name} with args: {prompt_args}")

            # Get the prompt from the MCP server
            prompt_messages = await self.mcp_client.get_prompt(prompt_name, prompt_args)

            if not prompt_messages:
                return f"Prompt '{prompt_name}' returned no messages."

            # Convert MCP prompt messages to chat messages and send to LLM
            chat_messages = []
            for msg in prompt_messages:
                if hasattr(msg, 'role') and hasattr(msg, 'content'):
                    if hasattr(msg.content, 'text'):
                        content = msg.content.text
                    else:
                        content = str(msg.content)

                    chat_messages.append({
                        "role": msg.role,
                        "content": content
                    })

            # If we have messages, send them to the chat client with tools
            if chat_messages:
                # Get tools for the prompt execution
                tools = self._mcp_cache.get("tools") or []
                if not tools and self.mcp_client:
                    try:
                        tools = await self.mcp_client.list_tools()
                    except Exception:
                        tools = []

                # Convert MCP tools to Azure AI tool format
                azure_tools = []
                for tool in tools:
                    tool_def = {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": {
                                "type": "object",
                                "properties": {},
                                "required": []
                            }
                        }
                    }

                    # Convert MCP tool input schema to Azure AI format
                    if hasattr(tool, 'inputSchema') and tool.inputSchema:
                        if isinstance(tool.inputSchema, dict):
                            if 'properties' in tool.inputSchema:
                                tool_def["function"]["parameters"]["properties"] = tool.inputSchema['properties']
                            if 'required' in tool.inputSchema:
                                tool_def["function"]["parameters"]["required"] = tool.inputSchema['required']
                        else:
                            # Handle case where inputSchema is an object with attributes
                            if hasattr(tool.inputSchema, 'properties') and tool.inputSchema.properties:
                                tool_def["function"]["parameters"]["properties"] = tool.inputSchema.properties
                            if hasattr(tool.inputSchema, 'required') and tool.inputSchema.required:
                                tool_def["function"]["parameters"]["required"] = tool.inputSchema.required

                    azure_tools.append(tool_def)

                response = await self.chat_client.chat_with_tools(chat_messages, azure_tools, self.mcp_client)

                # Add the prompt request and response to conversation history
                self.conversation_history.append({
                    "role": "user",
                    "content": f"Used prompt '{prompt_name}' with args: {prompt_args}"
                })
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response
                })

                return response
            else:
                return f"Prompt '{prompt_name}' did not generate any valid messages."

        except Exception as e:
            return f"Error using prompt: {e}"

    async def run(self):
        """Run the interactive CLI."""
        print("🤖 Interactive MCP Client with Azure AI / GitHub Models")
        print("=" * 60)

        if not await self.initialize():
            print("Failed to initialize clients. Exiting.")
            return

        await self.show_mcp_capabilities()

        print("\n💬 Start chatting! (Type 'quit', 'exit', or 'help' for commands)")
        print("📝 Use /prompt_name to trigger MCP prompts:")
        print("   • Positional args: /format document.pdf")
        print("   • Named args: /format doc_id=document.pdf")
        print("-" * 60)
        # Create an async prompt session with MCP autocompletion
        completer = MCPCompleter(self.mcp_client)
        # populate cache before first prompt
        await completer.refresh_once()
        # start background refresh task
        refresh_task = asyncio.create_task(completer.periodic_refresh())
        # start background MCP cache refresh task
        cache_refresh_task = asyncio.create_task(
            self._periodic_cache_refresh())

        session = PromptSession(completer=completer)

        try:
            while True:
                try:
                    # patch_stdout to allow prints while prompt is active
                    with patch_stdout():
                        user_input = (await session.prompt_async("\n👤 You: ")).strip()

                    if user_input.lower() in ['quit', 'exit', 'bye']:
                        print("👋 Goodbye!")
                        break
                    elif user_input.lower() == 'help':
                        await self.show_mcp_capabilities()
                        print("\n🆘 Additional Help:")
                        print("  • Use /prompt_name to trigger MCP prompts")
                        print("  • Positional args: /format document.pdf")
                        print("  • Named args: /format doc_id=document.pdf")
                        print("  • Mixed args: /format document.pdf style=formal")
                        print("  • Type 'clear' to clear conversation history")
                        print("  • Type 'quit', 'exit', or 'bye' to exit")
                        continue
                    elif user_input.lower() == 'clear':
                        self.conversation_history.clear()
                        print("🗑️ Conversation history cleared.")
                        continue
                    elif not user_input:
                        continue

                    print("🤖 Assistant: ", end="")
                    response = await self.handle_user_input(user_input)
                    print(response)

                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                except Exception as e:
                    print(f"\n❌ Error: {e}")
        finally:
            # cancel background refresh
            refresh_task.cancel()
            cache_refresh_task.cancel()
            try:
                await refresh_task
            except asyncio.CancelledError:
                pass
            try:
                await cache_refresh_task
            except asyncio.CancelledError:
                pass

            # Cancel any outstanding expansion task and await its completion
            try:
                if completer._expansion_task:
                    completer._expansion_task.cancel()
                    try:
                        await completer._expansion_task
                    except asyncio.CancelledError:
                        pass
            except Exception:
                pass

            # cleanup MCP client session
            try:
                if self.mcp_client:
                    await self.mcp_client.cleanup()
            except Exception:
                pass


class ChatClient:
    """Async wrapper around Azure ChatCompletionsClient for GitHub models."""

    def __init__(self, model: str = GITHUB_MODEL, token: str | None = None, endpoint: str | None = None):
        self._model = model
        token = token or os.environ.get("GITHUB_TOKEN")
        if not token:
            raise RuntimeError(
                "GITHUB_TOKEN not set. Provide token via env or pass `token=` to ChatClient.")

        endpoint = endpoint or os.environ.get(
            "GITHUB_MODELS_ENDPOINT", GITHUB_MODELS_ENDPOINT)
        self._client = ChatCompletionsClient(
            endpoint=endpoint, credential=AzureKeyCredential(token))

    async def chat(self, messages: Sequence[dict[str, str]]) -> str:
        """Send a chat-style message list to the model and return the assistant text."""
        return await self.chat_with_tools(messages, [])

    async def chat_with_tools(self, messages: Sequence[dict[str, str]], tools: list = None, mcp_client=None) -> str:
        """Send a chat-style message list with tools to the model and return the assistant text."""

        # Convert to the SDK message objects
        sdk_messages = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                sdk_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                sdk_messages.append(AssistantMessage(content=content))
            elif role == "tool":
                # Handle tool response messages
                tool_call_id = m.get("tool_call_id", "")
                sdk_messages.append(ToolMessage(
                    content=content, tool_call_id=tool_call_id))
            else:
                sdk_messages.append(UserMessage(content=content))

        loop = asyncio.get_event_loop()

        def _sync_call():
            try:
                if tools:
                    resp = self._client.complete(
                        messages=sdk_messages,
                        model=self._model,
                        tools=tools
                    )
                else:
                    resp = self._client.complete(
                        messages=sdk_messages,
                        model=self._model
                    )

                # Handle tool calls if present
                choice = resp.choices[0]
                if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                    return choice.message  # Return the whole message for tool call handling
                else:
                    return choice.message.content or str(resp)

            except Exception as e:
                print(f"Error in chat completion: {e}")
                try:
                    return resp.output[0].content[0].text
                except Exception:
                    return json.dumps(resp, default=str)

        result = await loop.run_in_executor(None, _sync_call)

        # If we got a message with tool calls, handle them
        if hasattr(result, 'tool_calls') and result.tool_calls and mcp_client:
            return await self._handle_tool_calls(result, sdk_messages, tools, mcp_client)

        return result if isinstance(result, str) else str(result)

    async def _handle_tool_calls(self, message, original_messages, tools, mcp_client):
        """Handle tool calls from the LLM response."""
        tool_messages = []

        for tool_call in message.tool_calls:
            try:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                print(f"🔧 Calling tool: {tool_name} with args: {tool_args}")

                # Call the MCP tool
                result = await mcp_client.call_tool(tool_name, tool_args)

                if result and result.content:
                    tool_result = ""
                    for content in result.content:
                        if hasattr(content, 'text'):
                            tool_result += content.text + "\n"
                        elif hasattr(content, 'data'):
                            tool_result += str(content.data) + "\n"
                else:
                    tool_result = f"Tool {tool_name} was called but returned no result."

                # Create tool message
                tool_messages.append({
                    "role": "tool",
                    "content": tool_result.strip(),
                    "tool_call_id": tool_call.id
                })

            except Exception as e:
                print(f"Error calling tool {tool_call.function.name}: {e}")
                tool_messages.append({
                    "role": "tool",
                    "content": f"Error calling tool: {e}",
                    "tool_call_id": tool_call.id
                })

        # Add the assistant message with tool calls and tool responses
        updated_messages = list(original_messages)
        updated_messages.append(AssistantMessage(
            content=message.content or "", tool_calls=message.tool_calls))

        # Add tool response messages
        for tool_msg in tool_messages:
            updated_messages.append(ToolMessage(
                content=tool_msg["content"],
                tool_call_id=tool_msg["tool_call_id"]
            ))

        # Get final response from the model
        loop = asyncio.get_event_loop()

        def _sync_call():
            try:
                resp = self._client.complete(
                    messages=updated_messages, model=self._model)
                return resp.choices[0].message.content or "The tool was executed successfully."
            except Exception as e:
                print(f"Error in final LLM call: {e}")
                return f"Tool executed successfully, but error in final response: {e}"

        return await loop.run_in_executor(None, _sync_call)


async def main():
    """Main entry point for the interactive MCP client."""
    client = InteractiveMCPClient()
    await client.run()


if __name__ == "__main__":
    # Allow correct event loop on Windows when needed; harmless on Linux.
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
