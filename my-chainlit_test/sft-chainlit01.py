# chainlit + Azure OpenAIのサンプル　元ファイルをAzure用に修正。2026/2/12
# chainlit run sft-chainlit01.py -w
import os
import chainlit as cl
# 1. Azure用のクラスをインポート
from openai import AsyncAzureOpenAI 
from dotenv import load_dotenv

load_dotenv()

# 2. Azure OpenAI クライアントの初期化
client = AsyncAzureOpenAI(
    api_key=os.getenv("API_KEY"),  # AzureのAPIキー
    azure_endpoint=os.getenv("API_ENDPOINT"),  # 例: https://xxx.openai.azure.com
    api_version=os.getenv("API_VERSION"),  # または "2023-05-15" など
)

@cl.on_chat_start
async def start():
    # チャット開始時に履歴を初期化
    cl.user_session.set("message_history", [
        {"role": "system", "content": "あなたは親切なアシスタントです。"}
    ])

@cl.on_message
async def main(message: cl.Message):
    # セッションから履歴を取得
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})

    # 応答メッセージの枠を作成
    msg = cl.Message(content="")
    
    # OpenAIのストリーミングレスポンス
    stream = await client.chat.completions.create(
        model="gpt-4o", # または gpt-4o-mini
        messages=message_history,
        stream=True
    )

# 3. 空の場合はスキップし、stream tokenを呼び出さないように修正
    async for part in stream:
        # choices が存在し、かつ中身があるかを確認
        if part.choices and len(part.choices) > 0:
            token = part.choices[0].delta.content or ""
            if token:
                await msg.stream_token(token)

    # 履歴に応答を追加して保存
    message_history.append({"role": "assistant", "content": msg.content})
    cl.user_session.set("message_history", message_history)
    await msg.send()