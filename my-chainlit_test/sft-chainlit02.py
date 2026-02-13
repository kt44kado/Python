# chainlit + AutoGen(Azure OpenAI)のサンプル
# chainlit run sft-chainlit02.py -w
import os
import chainlit as cl
from autogen_agentchat.agents import AssistantAgent
# 1. Azure用のClientに差し替え
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient 
from dotenv import load_dotenv

load_dotenv()

@cl.on_chat_start
async def setup_agent():
    # 1. モデルクライアントを一度だけ作成（キャッシュの役割）
    # これを関数の外（グローバル）に出しても良いですが、
    # セッションごとに設定を変えたい場合はここで行います。
    model_client = AzureOpenAIChatCompletionClient(
        model=os.getenv("DEPLOYMENT_NAME"),          # Azureポータルでのデプロイ名
        api_key=os.getenv("API_KEY"),
        azure_endpoint=os.getenv("API_ENDPOINT"),
        api_version=os.getenv("API_VERSION"),  # またはお使いのバージョン
    )
    # 2. エージェントの作成（model_clientを内部で保持・キャッシュ）
    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message="あなたは親切なAIアシスタントです。"
    )
    # 3. セッションに保存
    cl.user_session.set("agent", agent)

@cl.on_message
async def run_conversation(message: cl.Message):
    # 4. キャッシュされたエージェントを取り出す
    agent = cl.user_session.get("agent")

    # エージェントを実行（履歴などはagent内部のmodel_clientが管理）
    result = await agent.run(task=message.content)

    # 応答を送信
    await cl.Message(content=result.messages[-1].content, author="Assistant").send()