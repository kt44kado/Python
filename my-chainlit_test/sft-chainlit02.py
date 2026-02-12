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
    # モデルクライアントの設定
    model_client = AzureOpenAIChatCompletionClient(
        model=os.getenv("DEPLOYMENT_NAME"),          # Azureポータルでのデプロイ名
        api_key=os.getenv("API_KEY"),
        azure_endpoint=os.getenv("API_ENDPOINT"),
        api_version=os.getenv("API_VERSION"),  # またはお使いのバージョン
    )
    # エージェントの定義
    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message="あなたは親切なAIアシスタントです。"
    )

    cl.user_session.set("agent", agent)

@cl.on_message
async def run_conversation(message: cl.Message):
    agent = cl.user_session.get("agent")

    # ユーザーのメッセージを投げて応答を取得
    result = await agent.run(task=message.content)

    # 最新のメッセージ内容を送信
    await cl.Message(content=result.messages[-1].content, author="Assistant").send()