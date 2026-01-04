# プロンプトのテスト用バージョン
import os
import json
import pandas as pd
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

def format_mcp_notion_data(msg):
    """
    MCPの戻り値(msg)からNotionの検査データを抽出し、整形されたDataFrameを返す
    """
    # 1. msgからJSON文字列を抽出して辞書に変換
    # msg['content']はリストなので、最初の要素の'text'を取得
    try:
        raw_json_str = msg['content'][0]['text']
        data = json.loads(raw_json_str)
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        return f"データの解析に失敗しました: {e}"

    def get_v(props, name):
        """Notionのプロパティ型に応じて値を取り出すヘルパー"""
        p = props.get(name, {})
        p_type = p.get('type')
        
        if p_type == 'number':
            return p.get('number')
        if p_type == 'formula':
            return p.get('formula', {}).get('string')
        if p_type == 'date':
            return p.get('date', {}).get('start')
        if p_type == 'title':
            titles = p.get('title', [])
            return titles[0].get('plain_text') if titles else ""
        return None

    # 2. 各ページ(行)のデータを整形
    rows = []
    for page in data.get('results', []):
        props = page.get('properties', {})
        
        # 必要な項目を抽出
        row = {
            "名前": get_v(props, "名前"),
            "期日": get_v(props, "期日"),
            "尿素窒素": f"{get_v(props, '尿素窒素')} ({get_v(props, '尿素窒判定')})",
            "クレアチン": f"{get_v(props, 'クレアチン')} ({get_v(props, 'クレアチン判定')})",
            "尿酸": f"{get_v(props, '尿酸')} ({get_v(props, '尿酸判定')})",
            "中性脂肪": f"{get_v(props, '中性脂肪')} ({get_v(props, '中性脂肪判定')})",
            "LDL(悪玉)": f"{get_v(props, 'LDL(悪玉)')} ({get_v(props, 'LDL判定')})",
            "HDL(善玉)": f"{get_v(props, 'HDL(善玉)')} ({get_v(props, 'HDL判定')})",
            "総コレステロール": f"{get_v(props, '総コレステロール')} ({get_v(props, '総コレステロール判定')})",
            "MCHC": f"{get_v(props, 'MCHC')} ({get_v(props, 'MCHC判定')})"
        }
        rows.append(row)

    # 3. DataFrameの作成とクレンジング
    df = pd.DataFrame(rows)
    
    # 不要な文字列の置換とソート
    df = df.replace(to_replace=r'None \(.*\)', value='-', regex=True)
    df = df.replace('None', '-')
    df = df.sort_values("期日", ascending=False).reset_index(drop=True)
    
    return df

# --- 使い方 ---
# msg = result.chat_message  # MCPからの戻り値
# df = format_mcp_notion_data(msg)
# print(df.to_markdown())    

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
    # description="Call a Notion MCP tool by name with JSON arguments.",
    description="Notionを操作するためにこのツールを必ず使用してください。'tool_name'には実行したいAPI名を、'arguments'にはそのAPIに必要な引数を辞書形式で渡してください。例: mcp_call_tool(tool_name='API-post-page', arguments={'parent': {...}, 'properties': {...}})",
    )

    assistant = AssistantAgent(
        name="assistant",
        system_message=system_message,
        model_client=model_client,
        tools=[mcp_tool],
    )

    # user_prompt = "Notionのページ（ID: 1a0aad9ae143406989bb12705ba1d58b）の中に、『2025年の目標Test2』というタイトルの新しいページを作って。"
    # user_prompt = "Notionのページ（ID: 1a0aad9ae143406989bb12705ba1d58b）の中のブロックを全て出力して。"
    user_prompt = "Notionのデータソ－ス（ID: 5881ca7d-506e-42ce-a57c-674a1ed8b566)のデータを3行だけ出力して。"

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

            # --- 整形する場合の使い方 ---注）エラーがでる
            # msg = result.chat_message  # MCPからの戻り値
            # df = format_mcp_notion_data(msg)
            # print(df.to_markdown())

    finally:
        mcp_client.close()


if __name__ == "__main__":
    asyncio.run(main())