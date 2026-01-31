# Notion 対話型エージェントバージョン　無回答対策バージョン

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


def format_tools_for_prompt(mcp_tools) -> str:
    lines = []
    for t in mcp_tools:
        lines.append(
            f"- name: {t.name}\n"
            f"  description: {t.description}\n"
            f"  inputSchema: {json.dumps(t.inputSchema, ensure_ascii=False)}\n"
        )
    return "\n".join(lines)


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
    tools_catalog = format_tools_for_prompt(mcp_client.tools)
#    system_message = f"You are an assistant that manipulates Notion via Notion MCP tools.\n"
#    "You MUST call the tool `mcp_call_tool(tool_name, arguments)` to execute actions.\n"
#    "Choose tool_name from the catalog and pass arguments matching inputSchema.\n\n"
#    "MCP tool catalog:\nCatalog:\n{tools_catalog}"

    system_message = (
    "You are a Notion expert assistant equipped with MCP tools.\n"
    "Your goal is to fulfill user requests by efficiently managing Notion content.\n\n"
    
    "## Operational Guidelines:\n"
    "1. **ID-First Approach**: Always use unique IDs (e.g., page_id, database_id) for operations. "
    "If an ID is not provided, use the `search` tool to find the correct entity first.\n"
    "2. **Chain of Thought**: Before calling a tool, briefly analyze the necessary steps. "
    "For complex tasks (e.g., 'Move this task to the Done database'), search for both the item and the target database first.\n"
    "3. **Error Handling**: If a tool call fails due to '404 Not Found' or 'Unauthorized', "
    "explain to the user that the integration may lack access to that specific page and ask them to 'Share' it with the integration.\n"
    "4. **Data Integrity**: When creating or updating content, ensure all required properties match the schema provided in the tool catalog.\n\n"
    
    f"## MCP Tool Catalog:\n{tools_catalog}"
    )

    
    # model_client = OpenAIChatCompletionClient(model="gpt-5-mini")

    # ツール関数定義
    def mcp_call_tool(tool_name: str, arguments: dict) -> dict:
        result = mcp_client.call_tool(tool_name, arguments)
        # JSON変換ロジック...
        return result

    mcp_tool = FunctionTool(
        mcp_call_tool,
        name="mcp_call_tool",
        description="Notionを操作するためにこのツールを必ず使用してください。'tool_name'には実行したいAPI名を、'arguments'にはそのAPIに必要な引数を辞書形式で渡してください。例: mcp_call_tool(tool_name='API-post-page', arguments={'parent': {...}, 'properties': {...}})",
    )

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
        st.session_state.agent = AssistantAgent(
            name="assistant",
            model_client=client, # ここにClaude用のクライアントを渡す
            system_message=system_message,
            tools=[mcp_tool],
        )
        return st.session_state.agent
    
#    assistant = AssistantAgent(
#        name="assistant",
#        system_message=system_message,
#        llm_config={
#            "config_list": config_list,
#            "temperature": 0.7,
#        },
#        # model_client=model_client,
#        tools=[mcp_tool],
#    )
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
    # ユーザー入力を表示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        container = st.empty()
        full_response = ""
        
        # 1. run_stream の実行
        # インデントが async def の中にあることを確認してください
        async for chunk in assistant.run_stream(task=prompt):
            # TextMessage かつ 送信者がアシスタントであることを判定
            # chunk.content が空でないことも確認
            if isinstance(chunk, TextMessage) and chunk.source == assistant.name:
                if chunk.content:
                    full_response += chunk.content
                    container.markdown(full_response + "▌")
            
            # ツール実行エラーの判定（ModelAttribute 等の確認）
            elif hasattr(chunk, 'is_error') and chunk.is_error:
                error_msg = f"\n\n⚠️ **ツール実行エラー:** {chunk.content}"
                full_response += error_msg
                container.markdown(full_response)

        # 2. フォールバック処理
        # Notion操作成功後にテキストが空の場合、最後のメッセージから抽出を試みる
        if not full_response:
            full_response = "操作を完了しました。" # 暫定の成功メッセージ
        
        container.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})


# --- 5. 入力フォーム ---
if prompt := st.chat_input("メッセージを入力してください..."):
    # Streamlitの同期処理の中で非同期関数を実行する
    asyncio.run(run_chat(prompt))
