import os
from dotenv import load_dotenv
from notion_client import Client

# .envファイルを環境変数に読み込む
load_dotenv()
# クライアントの初期化
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
notion = Client(auth=NOTION_TOKEN)
DATABASE_ID = "c74d2c387135487ba14049d3daaab4ee"

from notion_client import Client
results = notion.databases.retrieve(DATABASE_ID) # DB本体のメタ情報を取得
results = notion.data_sources.query(results["data_sources"][0]["id"])
 # DBメタ情報の１行目のデータ－スからクエリーを実行して内容を取得する
 # print(results)

for page in results["results"]: # 一行ずつ変数pageに入れて処理を繰り返す
    if page["properties"]["名前"]["type"] == "title": # 名前がtilte型か確認
        title_data = page["properties"]["名前"]["title"] # 構造体である名前を取得
        if title_data and len(title_data) > 0: # 名前が空欄でないか確認
            name = title_data[0]["plain_text"] # 最初の要素の純粋な文字列を取得       
        
    print(f"{name}") # print(f"取得した名前は {name} です")のように書ける
