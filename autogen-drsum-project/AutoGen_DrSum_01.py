import asyncio
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.tools import FunctionTool

# Azure OpenAI 用（autogen-ext）
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient


# ====== 設定 ======
MCP_SERVER_DIR = r"c:\drsum-mcp-server"

# ★ここをあなたの Dr.Sum MCP Server の起動方法に合わせて修正してください
MCP_SERVER_ENTRY = ["python", r"server.py"]

TARGET_DATABASE: Optional[str] = None  # 固定するなら "SalesDB" など


class MCPBridge:
    """MCP stdio サーバーへ接続して、list_tools / call_tool を提供する薄いブリッジ。"""

    def __init__(self, server_params: StdioServerParameters):
        self._server_params = server_params
        self._session: Optional[ClientSession] = None
        self._stack = None

    async def __aenter__(self):
        self._stack = stdio_client(self._server_params)
        read, write = await self._stack.__aenter__()

        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._session:
            await self._session.__aexit__(exc_type, exc, tb)
        if self._stack:
            await self._stack.__aexit__(exc_type, exc, tb)

    async def list_tools(self) -> Dict[str, Any]:
        assert self._session is not None
        tools = await self._session.list_tools()
        return {
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.inputSchema,
                }
                for t in tools.tools
            ]
        }

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        assert self._session is not None
        res = await self._session.call_tool(name, arguments)
        return res.model_dump()


async def main():
    # .env を環境変数へ読み込み（カレントディレクトリの .env を想定）
    load_dotenv()

    deployment_name = os.getenv("DEPLOYMENT_NAME")
    api_key = os.getenv("API_KEY")
    api_endpoint = os.getenv("API_ENDPOINT")
    api_version = os.getenv("API_VERSION")

    missing = [k for k, v in {
        "DEPLOYMENT_NAME": deployment_name,
        "API_KEY": api_key,
        "API_ENDPOINT": api_endpoint,
        "API_VERSION": api_version,
    }.items() if not v]
    if missing:
        raise RuntimeError(f".env の値が不足しています: {missing}")

    # Dr.Sum MCP サーバー起動パラメータ
    env = os.environ.copy()
    server_params = StdioServerParameters(
        command=MCP_SERVER_ENTRY[0],
        args=MCP_SERVER_ENTRY[1:],
        cwd=MCP_SERVER_DIR,
        env=env,
    )

    async with MCPBridge(server_params) as mcp:

        async def mcp_list_tools() -> Dict[str, Any]:
            """MCP サーバーが提供するツール一覧を返す。"""
            return await mcp.list_tools()

        async def mcp_call_tool(tool_name: str, tool_args: Dict[str, Any]) -> Any:
            """任意の MCP ツールを tool_name/args で呼び出す。"""
            return await mcp.call_tool(tool_name, tool_args)

        tools = [
            FunctionTool(
                mcp_list_tools,
                name="mcp_list_tools",
                description="List available tools exposed by the Dr.Sum MCP server.",
            ),
            FunctionTool(
                mcp_call_tool,
                name="mcp_call_tool",
                description=(
                    "Call an MCP tool by name with JSON arguments. "
                    "Use after discovering tool names via mcp_list_tools."
                ),
            ),
        ]

        # Azure OpenAI（.env の値を使用）
        model_client = AzureOpenAIChatCompletionClient(
            azure_endpoint=api_endpoint,
            api_key=api_key,
            api_version=api_version,
            azure_deployment=deployment_name,
            # model は論理名として必要になることがあります。通常はベースモデル名か deployment 名を指定。
            model="gpt-4o",
            temperature=0,
        )

        system_message = """You are an assistant that explores Dr.Sum via a local MCP server.
First call mcp_list_tools, then use mcp_call_tool to:
- list databases
- list tables and views in the target database
- fetch sample data (top N rows) for a few objects
Return results in structured JSON and short explanation in Japanese.
If a required tool does not exist, explain what tool is missing based on mcp_list_tools output.
"""

        assistant = AssistantAgent(
            name="drsum_assistant",
            model_client=model_client,
            tools=tools,
            system_message=system_message,
        )

        if TARGET_DATABASE:
            task = f"""
Dr.Sum を MCP 経由で調査してください。
対象DB: {TARGET_DATABASE}

手順:
1) mcp_list_tools でツールを確認
2) DB一覧取得
3) 対象DBのテーブル一覧・ビュー一覧を取得
4) 代表的なテーブル/ビューを2〜3個選び、各オブジェクトから先頭5行を取得（もしくは rowcount）
結果を JSON でまとめてください。
"""
        else:
            task = """
Dr.Sum を MCP 経由で調査してください。

手順:
1) mcp_list_tools でツールを確認
2) DB一覧取得
3) 最初のDBを選び、そのDBのテーブル一覧・ビュー一覧を取得
4) 代表的なテーブル/ビューを2〜3個選び、各オブジェクトから先頭5行を取得（もしくは rowcount）
結果を JSON でまとめてください。
"""

        await Console(assistant.run_stream(task=task))


if __name__ == "__main__":
    asyncio.run(main())