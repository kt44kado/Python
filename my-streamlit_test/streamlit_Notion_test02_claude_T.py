# Notion 対話型エージェントバージョン　test2のClaude版 & トークン削減版

import os
import json
import threading

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import streamlit as st
import asyncio

from autogen_core import CancellationToken
from autogen_core.tools import FunctionTool

from autogen_agentchat.agents import AssistantAgent
# from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.agents import AssistantAgent


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
            env={**os.environ, 
                 "NOTION_API_KEY": self.notion_api_key
                },
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


# ツール一覧は “名前と短い説明だけにして、schema は必要時にだけ参照する方式にします。
def format_tools_for_prompt_min(mcp_tools) -> str:
    return "\n".join([f"- {t.name}: {t.description}" for t in mcp_tools])

st.set_page_config(page_title="AutoGen x Streamlit App", layout="centered")
st.title("🤖 AutoGen 対話型エージェント")

# --- 1. エージェントの初期化（キャッシュを利用） ---
@st.cache_resource
def get_mcp_client():
    notion_token = st.secrets["NOTION_TOKEN"] # StreamlitではSecrets管理を推奨
    client = McpNotionClient(notion_api_key=notion_token)
    client.start()  # 接続開始
    return client

# --- キャッシュ対象2: エージェントとツールの構築 ---
@st.cache_resource
def get_assistant():
    mcp_client = get_mcp_client() # キャッシュされたクライアントを取得
    
    # ツールカタログの作成
#    tools_catalog = format_tools_for_prompt(mcp_client.tools)
    tools_catalog = format_tools_for_prompt_min(mcp_client.tools)

    system_message = (
    "You are a Notion assistant equipped with MCP tools.\n"
    "Use the tool mcp_call_tool.\n"
    "If you need a tool's arguments schema, ask the user for missing info or call get_tool_schema.\n"
    f"\n## Tool List:\n{tools_catalog}"
    )

# Notionの tool result を「モデルに渡す前に圧縮」する
    def compact_notion_result(tool_name: str, raw):
        # raw が dict だと仮定（実際の型に応じて調整）
        if isinstance(raw, dict):
            # 例: ページ作成結果なら最低限だけ返す
            if "page" in tool_name.lower() or "post-page" in tool_name.lower() or "create" in tool_name.lower():
                return {
                    "id": raw.get("id"),
                    "url": raw.get("url"),
                    "created_time": raw.get("created_time"),
                }
            # search なら候補だけ
            if "search" in tool_name.lower():
                results = raw.get("results", [])
                slim = []
                for r in results[:5]:
                    slim.append({
                        "id": r.get("id"),
                        "object": r.get("object"),
                        "url": r.get("url"),
                    })
                return {"results": slim, "has_more": raw.get("has_more")}
        return raw  # 最後の保険（本当はここも絞りたい）

    # ツール関数定義
    def mcp_call_tool(tool_name: str, arguments: dict) -> dict:
    #    result = mcp_client.call_tool(tool_name, arguments)
    #    # JSON変換ロジック...
    #    return result
        raw = mcp_client.call_tool(tool_name, arguments)
        return compact_notion_result(tool_name, raw)

    mcp_tool = FunctionTool(
        mcp_call_tool,
        name="mcp_call_tool",
        description="Notionを操作するためにこのツールを必ず使用してください。'tool_name'には実行したいAPI名を、'arguments'にはそのAPIに必要な引数を辞書形式で渡してください。例: mcp_call_tool(tool_name='API-post-page', arguments={'parent': {...}, 'properties': {...}})",
    )

    # schema を必要時だけ返す軽いツールを追加
    tool_map = {t.name: t for t in mcp_client.tools}

    def get_tool_schema(tool_name: str) -> dict:
        t = tool_map.get(tool_name)
        if not t:
            return {"error": "unknown tool"}
        return {"name": t.name, "description": t.description, "inputSchema": t.inputSchema}

    schema_tool = FunctionTool(get_tool_schema, name="get_tool_schema",
        description="Return schema for a specific MCP tool.")

    # エージェント作成
    def get_model_client():
        # st.secretsからAPIキーを取得して渡す
        api_key = st.secrets["ANTHROPIC_API_KEY"]
    
        return AnthropicChatCompletionClient(
        #    model="claude-sonnet-4-20250514",
            model="claude-sonnet-4-5-20250929",
        #    model="claude-haiku-4-5-20251001",
            api_key=api_key, # 明示的に指定
            temperature=0.7,
        )
    if "agent" not in st.session_state:
        client = get_model_client()
#        client._model = "claude-haiku-4-5-20251001"
        st.session_state.agent = AssistantAgent(
            name="assistant",
            model_client=client, # ここにClaude用のクライアントを渡す
            system_message=system_message,
        #    tools=[mcp_tool],
            tools=[mcp_tool, schema_tool]
        )
        return st.session_state.agent
    
    return assistant

# --- メイン処理 ---
# --- 1. エージェントの初期化（キャッシュを利用） ---
assistant = get_assistant()

# --- 2. セッション状態の初期化（会話履歴の保存用） ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. 保存された履歴の表示 ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 4. メイン処理（非同期関数として定義） ---
async def run_chat(prompt):
    # ユーザー入力を画面に表示 & 履歴に追加
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # アシスタントの応答領域を作成
    with st.chat_message("assistant"):
        container = st.empty()  # ストリーミング表示用の空枠
        full_response = ""
        
        # run_stream を使用して逐次取得
        # ※ 実際の実装では TaskResult が流れてくるため、それを取り出す
        async for chunk in assistant.run_stream(task=prompt):
            # 1. 通常のテキスト回答の処理
            if isinstance(chunk, TextMessage) and chunk.source == assistant.name:
                full_response += chunk.content
                container.markdown(full_response + "▌") # カーソル風の演出
            
            # 2. ツール実行エラーの処理を追加
            elif hasattr(chunk, 'is_error') and chunk.is_error:
                error_msg = f"\n\n⚠️ **ツール実行エラー:** {chunk.content}"
                full_response += error_msg
                container.markdown(full_response)
        
        # 最終結果の表示（何も返ってこなかった場合のフォールバック）
        if not full_response:
            full_response = "申し訳ありません。回答を生成できませんでした（ツール実行に失敗した可能性があります）。"
        
        container.markdown(full_response) # 最終結果を確定表示
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- 5. 入力フォーム ---
if prompt := st.chat_input("メッセージを入力してください..."):
    # Streamlitの同期処理の中で非同期関数を実行する
    asyncio.run(run_chat(prompt))
