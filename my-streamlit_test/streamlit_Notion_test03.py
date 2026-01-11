# Notion 対話型エージェント　捏造対策バージョン 

import os
import json
import jsonschema
import threading

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import streamlit as st
import asyncio

from autogen_core import CancellationToken
from autogen_core.tools import FunctionTool

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.agents import AssistantAgent
from typing import Any, Optional

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

def _json_dump_if_needed(x):
    # dict/list を JSON文字列に
    if isinstance(x, (dict, list)):
        return json.dumps(x, ensure_ascii=False)
    return x

def normalize_mcp_args(tool_name: str, args: dict) -> dict:
    """
    notion-mcp-server のスキーマ都合で
    icon/cover が "string(format: json)"、children が "array of string" になっているケースを吸収。
    """
    if not isinstance(args, dict):
        return args

    # API-post-page の典型：icon/cover は JSON 文字列になっていることがある
    if tool_name == "API-post-page":
        if "icon" in args:
            args["icon"] = _json_dump_if_needed(args["icon"])
        if "cover" in args:
            args["cover"] = _json_dump_if_needed(args["cover"])

        # children: schemaだと items が string なので、dict のまま渡すと弾かれる可能性がある
        if "children" in args and isinstance(args["children"], list):
            args["children"] = [_json_dump_if_needed(b) for b in args["children"]]

    return args

def make_mcp_function_tool(mcp_client, mcp_tool_def):
    input_schema = mcp_tool_def.inputSchema

    def _call(arguments: Optional[dict] = None, **kwargs: Any) -> dict:
        # LLMが arguments={...} で渡してきても、parent=... で渡してきても吸収
        payload = {}
        if isinstance(arguments, dict):
            payload.update(arguments)
        payload.update(kwargs)

        # 任意：前に入れた正規化（children/icon/cover 等）を使うならここで
        # payload = normalize_mcp_args(mcp_tool_def.name, payload)

        # 入力検証（入れているなら）
        jsonschema.validate(instance=payload, schema=input_schema)

        result = mcp_client.call_tool(mcp_tool_def.name, payload)

        try:
            return result.model_dump()
        except Exception:
            if hasattr(result, "content"):
                return {"content": result.content, "isError": getattr(result, "isError", False)}
            return {"result": result}

    return FunctionTool(
        _call,
        name=mcp_tool_def.name,          # 例: "API-post-page"
        description=mcp_tool_def.description,
    )

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
    mcp_client = get_mcp_client()

    # system_message が壊れていたので """ """ で確実に入れる
    tools_catalog = format_tools_for_prompt(mcp_client.tools)
    system_message = f"""You are an assistant that manipulates Notion via MCP tools.

RULES:
- To perform any Notion action, you MUST call one of the provided tools.
- NEVER invent tool names. Only call the tools provided to you.
- For each tool call, pass arguments that match the tool's inputSchema.

MCP tool catalog (reference):
{tools_catalog}
"""

    model_client = OpenAIChatCompletionClient(model="gpt-5-mini")

    notion_tools = [make_mcp_function_tool(mcp_client, t) for t in mcp_client.tools]

    assistant = AssistantAgent(
        name="assistant",
        system_message=system_message,
        model_client=model_client,
        tools=notion_tools,   # ← ここが重要：mcp_call_tool 1個ではなく全ツールを列挙
    )
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
