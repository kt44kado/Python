# OpenAI Platfrm ChatがNotion_MCP>pyを元に変換したコード
import os
import json
import threading
import asyncio
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import autogen


class McpNotionClient:
    """
    Notion MCP server をバックグラウンドスレッド(専用event loop)で起動し、
    同期関数から tool call できるようにするラッパー。
    """
    def __init__(self, notion_api_key: str):
        self.notion_api_key = notion_api_key

        self._thread = None
        self._loop = None

        self._ready = threading.Event()
        self._shutdown = None  # asyncio.Event (loop上で作る)
        self._session = None

        self.tools = []  # MCP tool list

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
        # runner側の await self._shutdown.wait() を解除
        asyncio.run_coroutine_threadsafe(self._async_shutdown(), self._loop).result(timeout=30)
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=30)

    def call_tool(self, tool_name: str, arguments: dict):
        """同期API: AutoGenのfunction executionから呼ぶ想定"""
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
    # LLMが「tool_name と arguments」を正しく作れるように、一覧をプロンプトへ渡す
    lines = []
    for t in mcp_tools:
        lines.append(
            f"- name: {t.name}\n"
            f"  description: {t.description}\n"
            f"  inputSchema: {json.dumps(t.inputSchema, ensure_ascii=False)}\n"
        )
    return "\n".join(lines)


def main():
    load_dotenv()
    notion_token = os.getenv("NOTION_TOKEN")  # あなたの元コードに合わせて NOTION_TOKEN を読む
    if not notion_token:
        raise RuntimeError("NOTION_TOKEN is not set in environment variables/.env")

    # MCP(Notion)を常駐起動
    mcp_client = McpNotionClient(notion_api_key=notion_token)
    print("Starting Notion MCP server...")
    mcp_client.start()
    print(f"Connected. MCP tools: {len(mcp_client.tools)}")

    tools_catalog = format_tools_for_prompt(mcp_client.tools)

    # AutoGen 設定（OpenAIキーは環境変数から）
    llm_config = {
        "config_list": [
            {
                "model": "gpt-4o",
                # api_key は OPENAI_API_KEY 環境変数から自動で読ませる想定
            }
        ],
        "temperature": 0,
    }

    system_message = (
        "You are an assistant that manipulates Notion via Notion MCP tools.\n"
        "You must use the function `mcp_call_tool(tool_name, arguments)` to execute tools.\n"
        "Select the correct tool_name from the catalog and pass arguments that match inputSchema.\n\n"
        "MCP tool catalog:\n"
        f"{tools_catalog}\n"
    )

    assistant = autogen.AssistantAgent(
        name="assistant",
        llm_config=llm_config,
        system_message=system_message,
    )

    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        code_execution_config=False,
        max_consecutive_auto_reply=5,
    )

    # AutoGen に「実行できる関数」として登録（= LLMがfunction callingで呼べる）
    @user_proxy.register_for_execution()
    @assistant.register_for_llm(description="Call a Notion MCP tool by name with JSON arguments.")
    def mcp_call_tool(tool_name: str, arguments: dict) -> dict:
        """
        tool_name: MCPのツール名（catalogのname）
        arguments: inputSchemaに一致するJSON引数
        """
        result = mcp_client.call_tool(tool_name, arguments)
        # MCPの返り値は独自型の場合があるので、念のため dict/str に寄せる
        try:
            return json.loads(json.dumps(result, default=lambda o: getattr(o, "__dict__", str(o))))
        except Exception:
            return {"result": str(result)}

    # ユーザー指示（あなたの元コードと同等）
    user_prompt = "Notionのページ（ID: 1a0aad9ae143406989bb12705ba1d58b）の中に、『2025年の目標Test2』というタイトルの新しいページを作って。"

    try:
        user_proxy.initiate_chat(assistant, message=user_prompt)
    finally:
        mcp_client.close()


if __name__ == "__main__":
    main()