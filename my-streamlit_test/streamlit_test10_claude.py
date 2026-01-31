# 対話型エージェントバージョン
# 提案コードからの変更　１．modelをgpt-4oからgpt-5-miniに変更、２，AI回答の文頭から依頼分を削除
import streamlit as st
import asyncio
from autogen_agentchat.agents import AssistantAgent
#from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_agentchat.messages import TextMessage
import os
from dotenv import load_dotenv
load_dotenv()

#config_list = [
#    {
#        "model": "claude-sonnet-4-20250514",
#        "api_key": os.getenv("ANTHROPIC_API_KEY"),
#        "api_type": "anthropic"
#    }
#]

st.set_page_config(page_title="AutoGen x Streamlit App", layout="centered")
st.title("🤖 AutoGen 対話型エージェント")

# --- 1. エージェントの初期化（キャッシュを利用） ---
@st.cache_resource

def get_model_client():
    # 環境変数 ANTHROPIC_API_KEY が設定されている前提
    return AnthropicChatCompletionClient(
    #    model="claude-sonnet-4-20250514", # 任意のモデル
        model="claude-sonnet-4-5-20250929",
    #    model="claude-haiku-4-5-20251001",
        temperature=0.7,
    )

def get_agent():
    if "agent" not in st.session_state:
        client = get_model_client()
        st.session_state.agent = AssistantAgent(
            name="assistant",
            model_client=client, # ここにClaude用のクライアントを渡す
            system_message="あなたは有能なアシスタントです。簡潔で分かりやすい回答を心がけてください。"
        )
    return st.session_state.agent

#def get_agent():
    # 環境変数が設定済みである前提
    # model_client = OpenAIChatCompletionClient(model="gpt-5-mini")
    
    # エージェントの作成（ここでMCPツールなどを追加することも可能）

#    agent = AssistantAgent(
#        name="assistant",
#        llm_config={
#            "config_list": config_list,
#            "temperature": 0.7,
#        },
#        model_client=model_client,
#        system_message="あなたは有能なアシスタントです。簡潔で分かりやすい回答を心がけてください。"
#    )
#    return agent

agent = get_agent()

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
        async for chunk in agent.run_stream(task=prompt):
            # chunkの種類（思考中、ツール実行中、最終回答など）を判定
            # 下記は簡易的にテキストコンテンツを取得するイメージ
            # if hasattr(chunk, 'content') and chunk.content:
            # 上のコードを変更（文頭の入力文を削除）　2026年最新仕様の判定方法：
            # chunkが「TextMessage」であり、かつ送信元がエージェント自身（agent.name）である場合のみ採用する
            # これにより、入力プロンプト（userからのログ）が混ざるのを防ぎます
            if isinstance(chunk, TextMessage) and chunk.source == agent.name:
                full_response += chunk.content
                container.markdown(full_response + "▌") # カーソル風の演出
        
        container.markdown(full_response) # 最終結果を確定表示
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- 5. 入力フォーム ---
if prompt := st.chat_input("メッセージを入力してください..."):
    # Streamlitの同期処理の中で非同期関数を実行する
    asyncio.run(run_chat(prompt))
