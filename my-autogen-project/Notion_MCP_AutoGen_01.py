import sys
import os
from dotenv import load_dotenv
# from notion_client import Client # NotionSDKは使わない
import asyncio
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
# from requests import session
import json  # ← これを追加
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
from autogen_ext.tools.mcp import StdioServerParams
from autogen_ext.tools.mcp._factory import StdioMcpToolAdapter
# mcp 直接の ClientSession ではなく、AutoGenが期待する形式に合わせます
# 1. 必要なクラスを直接インポート（ImportErrorを避けるためこのパスで）
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console

# .envファイルからNotionキーを環境変数に読み込む（OpenAIキーは元々環境変数に保存済）
load_dotenv()
notion_token = os.getenv("NOTION_TOKEN")
# print(notion_token)

def get_model_client() -> OpenAIChatCompletionClient:
    return OpenAIChatCompletionClient(
        model="gpt-4o",
        # api_key=os.getenv("OPENAI_API_KEY")
        api_key = OpenAI() # APIキーは環境変数から自動的に取得される
    )

agent_prompt = """
ユーザーの指示に基づいて、提供されたツールを使用してNotion内のページやブロック、データベースの情報取得や、
Notionへのページやブロック、データベースへの追加を処理してください。
"""

async def main():
    # Notion MCPサーバーの起動設定 (npxを使用)
    server_params = StdioServerParameters(
        command="npx.cmd",
        args=["-y", "@notionhq/notion-mcp-server"],
        env={**os.environ, "NOTION_API_KEY": notion_token} # これに変更する
    )

    print("Notion MCPサーバーに接続中...")


    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as mcp_session:
            await mcp_session.initialize()
            print("Notion接続成功。ツールを変換中...")

            # 1. ツールリストを取得
            mcp_tools_list = await mcp_session.list_tools()
            
            # 2. 【最重要】引数名を付けず、順番通りに渡す
            # 現在の autogen-ext v0.4.x の多くでは、第一引数が session(client) です
            autogen_tools = []
            for t in mcp_tools_list.tools:
                try:
                    # 名前付き引数を使わず、位置引数で渡すことで TypeError を回避します
                    adapter = McpToolAdapter(mcp_session, t)
                    autogen_tools.append(adapter)
                except Exception as e:
                    print(f"ツールの変換に失敗しました ({t.name}): {e}")

            # 3. エージェントの作成
            notion_agent = AssistantAgent(
                name="notion_agent",
                model_client=get_model_client(),
                tools=autogen_tools,
                system_message=agent_prompt,
            )

            print(f"{len(autogen_tools)}個のツールを読み込みました。実行します。")
            await Console(notion_agent.run_stream(task="Notionの情報を確認して"))


if __name__ == "__main__":
    asyncio.run(main())
