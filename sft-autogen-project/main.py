import os
from dotenv import load_dotenv
from autogen_agentchat import BaseMessage

# 環境変数の読み込み
load_dotenv()

# メッセージの作成
message = BaseMessage()

# メッセージの送信
message_content = "こんにちは、これはテストメッセージです。"
print(message_content)
