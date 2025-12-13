import os
from openai import OpenAI

client = OpenAI() # 環境変数から自動読み込み

# 使用するモデルによっては、chat.completions ではなく completions を使う場合があります
# 今回のエラーメッセージはその可能性を示唆しています
model_to_use = "gpt-3.5-turbo-instruct" # 例: このモデルでエラーが出やすい

# プロンプトを定義
prompt_text = "PythonでOpenAI APIを使う方法を簡潔に教えてください。"

try:
    # APIリクエストを送信（completions APIの場合の例）
    response = client.completions.create(
        model=model_to_use,
        prompt=prompt_text,
        # ここを修正！
        max_tokens=500,        # 生成する最大トークン数
        temperature=0.7
    )

    # レスポンスからテキスト部分を抽出して表示
    if response.choices:
        print("AIからの回答:")
        print(response.choices[0].text.strip())
    else:
        print("回答が得られませんでした。")

except Exception as e:
    print(f"API呼び出し中にエラーが発生しました: {e}")