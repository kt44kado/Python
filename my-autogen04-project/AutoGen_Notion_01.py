import os
import json
import threading
import asyncio
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from autogen_core import CancellationToken
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage


class McpNotionClient:
    def __init__(self, notion_api_key: str):
        self.notion_api_key = notion_api_key
        self._thread = None
        self._loop = None
        self._ready = threading.Event()
        self._shutdown = None
        self._session = None
        self.tools = []

        self.server_params = StdioServerParameters(
            command="npx",
            args=["-y", "@notionhq/notion-mcp-server"],
            env={**os.environ, "NOTION_API_KEY": self.notion_api_key},
        )

    def start(self):
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=60)
        if not self._ready.is_set():
            raise RuntimeError("MCP Notion client failed to become ready in time.")

    def close(self):
        if not self._loop:
            return
        asyncio.run_coroutine_threadsafe(self._async_shutdown(), self._loop).result(timeout=30)
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=30)

    def call_tool(self, tool_name: str, arguments: dict):
        if not self._ready.is_set():
            raise RuntimeError("MCP Notion client is not ready yet.")
        coro = self._session.call_tool(name=tool_name, arguments=arguments)
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=60)

    def _thread_main(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.create_task(self._runner())
        self._loop.run_forever()

    async def _runner(self):
        self._shutdown = asyncio.Event()
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self._session = session
                resp = await session.list_tools()
                self.tools = resp.tools
                self._ready.set()
                await self._shutdown.wait()

    async def _async_shutdown(self):
        if self._shutdown:
            self._shutdown.set()


def format_tools_for_prompt(mcp_tools) -> str:
    lines = []
    for t in mcp_tools:
        lines.append(
            f"- name: {t.name}\n"
            f"  description: {t.description}\n"
            f"  inputSchema: {json.dumps(t.inputSchema, ensure_ascii=False)}\n"
        )
    return "\n".join(lines)


async def main():
    load_dotenv()

    notion_token = os.getenv("NOTION_TOKEN")
    if not notion_token:
        raise RuntimeError("NOTION_TOKEN is not set")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    mcp_client = McpNotionClient(notion_api_key=notion_token)
    print("Starting Notion MCP server...")
    mcp_client.start()
    print(f"Connected. MCP tools: {len(mcp_client.tools)}")

    tools_catalog = format_tools_for_prompt(mcp_client.tools)

    system_message = (
        "You are an assistant that manipulates Notion via Notion MCP tools.\n"
        "You MUST call the tool `mcp_call_tool(tool_name, arguments)` to execute actions.\n"
        "Choose tool_name from the catalog and pass arguments matching inputSchema.\n\n"
        "MCP tool catalog:\n"
        f"{tools_catalog}\n"
    )

    # OpenAI model client（環境変数 OPENAI_API_KEY を使用）
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    def mcp_call_tool(tool_name: str, arguments: dict) -> dict:
        result = mcp_client.call_tool(tool_name, arguments)
        try:
            return json.loads(json.dumps(result, default=lambda o: getattr(o, "__dict__", str(o))))
        except Exception:
            return {"result": str(result)}

    mcp_tool = FunctionTool(
    mcp_call_tool,  # ← fn= をやめて位置引数で渡す
    name="mcp_call_tool",
    description="Call a Notion MCP tool by name with JSON arguments.",
    )

    assistant = AssistantAgent(
        name="assistant",
        system_message=system_message,
        model_client=model_client,
        tools=[mcp_tool],
    )

    user_prompt = "Notionのページ（ID: 1a0aad9ae143406989bb12705ba1d58b）の中に、『2025年の目標Test2』というタイトルの新しいページを作って。"

    try:
        token = CancellationToken()
        result = await assistant.on_messages(
           [TextMessage(content=user_prompt, source="user")],
           cancellation_token=token,
        )
        # いまの result は Response(...) なので、中の chat_message を見る
        msg = result.chat_message

        print("type:", type(msg).__name__)

        # Tool実行結果（FunctionExecutionResult）の中身だけ取り出す
        if hasattr(msg, "results") and msg.results:
            # results[0].content は文字列（dictっぽい文字列）なのでそのまま出すか、JSONだけ抜く
            raw = msg.results[0].content
            print("tool result (raw):")
            print(raw)

            # もし Notion URL だけ取りたいなら、雑に "url":"..." を探す
            import re
            m = re.search(r'"url":"([^"]+)"', raw)
            if m:
                print("Notion URL:", m.group(1))
        else:
            # 通常会話のみの場合
            print("message:", getattr(msg, "content", msg))

    finally:
        mcp_client.close()


if __name__ == "__main__":
    asyncio.run(main())