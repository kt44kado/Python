# chainlit のサンプル
import os
import chainlit as cl
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# OpenAIクライアントの初期化
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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

    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await msg.stream_token(token)

    # 履歴に応答を追加して保存
    message_history.append({"role": "assistant", "content": msg.content})
    cl.user_session.set("message_history", message_history)
    await msg.send()