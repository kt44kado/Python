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
            # 修正前
            # mcp_tools = await session.list_tools()
            # print(f"取得したツール数: {len(mcp_tools)}")
            # 修正後
            response = await session.list_tools()
            mcp_tools = response.tools  # responseの中からtoolsリストを取り出す
            print(f"取得したツール数: {len(mcp_tools)}")

            # 41. OpenAIに渡す形式に変換（McpTool -> OpenAI Function Def）
            openai_tools_format = [{"type": "function", "function": tool.dict()} for tool in mcp_tools]

            # 5. ユーザーからの指示
            user_prompt = "Notionで『2025年の目標』というタイトルの新しいページを作って、内容は『MCPをマスターする』にして。"
            
            # 6. OpenAIにツール情報を渡して実行（Function Callingの仕組み）
            # 本来はループを回してツール実行結果を返しますが、ここでは概念を示します
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": user_prompt}],
                tools=openai_tools_format,
                tool_choice="auto", # OpenAIに実行する関数を選ばせる
            )
            # 5. OpenAIの判断を処理（Function Callingが発生した場合）
            first_choice = response.choices[0].message
            if first_choice.tool_calls:
                tool_call = first_choice.tool_calls[0]
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                print(f"OpenAIが関数呼び出しを提案: {function_name} with args {function_args}")

                # 6. MCPサーバー経由でNotionの機能を実行
                # 修正前
                # tool_response = await session.invoke_tool(
                #    name=function_name,
                #    parameters=function_args
                #)
                # 修正後
                tool_response = await session.call_tool(
                name=tool_call.function.name,
                arguments=function_args
                )
                print(f"Notionからの実行結果: {tool_response}")
            else:
                print("OpenAIは関数呼び出しを提案しませんでした。")

            print("AIの判断:", response.choices[0].message.tool_calls[0].function)
            # この後、AIが選んだツールをsession.call_tool()で実際に実行します

# 非同期メイン関数の実行
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
