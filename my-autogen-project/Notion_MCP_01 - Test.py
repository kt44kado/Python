import os
from dotenv import load_dotenv
# from notion_client import Client # NotionSDKは使わない
import asyncio
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from requests import session
import json  # ← これを追加

# .envファイルからNotionキーを環境変数に読み込む（OpenAIキーは元々環境変数に保存済）
load_dotenv()
notion_token = os.getenv("NOTION_TOKEN")
print(notion_token)

# 1. OpenAIの初期化
openai_client = OpenAI() # APIキーは環境変数から自動的に取得される

# 2. Notion MCPサーバーの起動設定 (npxを使用)
server_params = StdioServerParameters(
    command="npx",
    # args=["-y", "@modelcontextprotocol/server-notion"],
    args=["-y", "@notionhq/notion-mcp-server"],
    env={"NOTION_ACCESS_TOKEN": notion_token}
)
print("Notion MCPサーバーを起動中...")

async def main():
    # 3. MCPサーバーと通信開始
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Notion MCPサーバーと接続完了。")

            # 4. MCPサーバーが持っている「Notion操作ツール」のリストを取得
            response = await session.list_tools()
            mcp_tools = response.tools  # responseの中からtoolsリストを取り出す
            print(f"取得したツール数: {len(mcp_tools)}")

            # 5. MCPツールをOpenAI形式に変換
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema, # MCPのスキーマをそのまま利用可能
                },
            }
            for tool in mcp_tools
            ]
            # 6. OpenAI APIへのリクエスト
            messages = [
                 {"role": "system", "content": "あなたはNotion操作の専門家です。"},
                 {"role": "user", "content": "Notionで『今日のタスク』というタイトルのページを作成して"}
            ]
            response = await openai_client.chat.completions.create(
                model="gpt-4o", # または gpt-4-turbo 等
                messages=messages,
                tools=openai_tools,
                tool_choice="auto"
            )
            # 7. OpenAIからの応答（ツール実行の指示）を処理
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls
            if tool_calls:
                 # OpenAIがNotion操作が必要だと判断した場合
                 for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    print(f"ツール実行中: {function_name}({function_args})")
                    # 8. 実際にMCPサーバー経由でNotionを操作
                    # call_toolの戻り値はMCPサーバー側の実装に依存します
                    result = await session.call_tool(function_name, function_args)
        
                    # 実行結果をメッセージ履歴に追加して、最終的な返答を生成
                    messages.append(response_message)
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": str(result.content),
                    })

                 # 最終的な回答をOpenAIから取得
                 final_response = await openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                 )
                 print(final_response.choices[0].message.content)


