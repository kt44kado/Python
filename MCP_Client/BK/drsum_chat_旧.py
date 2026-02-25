"""
DrSum 自然言語チャットクライアント

Azure OpenAI の Function Calling を使い、自然言語の指示を
DrSum MCP Bridge Server の API 呼び出しに変換して実行する。

使い方:
  1. .env ファイルを作成して環境変数を設定
  2. pip install -r requirements.txt
  3. python drsum_chat.py
"""

import json
import os
import sys

import requests
from dotenv import load_dotenv
from openai import AzureOpenAI

# ── 環境変数読み込み ──────────────────────────────────
load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

BRIDGE_URL = os.getenv("BRIDGE_URL", "http://eaasys022s:3000")
DRSUM_USER = os.getenv("DRSUM_USER")
DRSUM_PASSWORD = os.getenv("DRSUM_PASSWORD")
DRSUM_HOST = os.getenv("DRSUM_HOST")
DRSUM_PORT = os.getenv("DRSUM_PORT")


# ── Azure OpenAI クライアント初期化 ──────────────────
def create_openai_client():
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
        print("エラー: AZURE_OPENAI_ENDPOINT と AZURE_OPENAI_API_KEY を .env に設定してください。")
        sys.exit(1)
    return AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )


# ── Bridge Server API 呼び出し関数 ───────────────────
def _build_auth():
    """共通の認証パラメータを返す"""
    auth = {"user": DRSUM_USER, "password": DRSUM_PASSWORD}
    if DRSUM_HOST:
        auth["host"] = DRSUM_HOST
    if DRSUM_PORT:
        auth["port"] = int(DRSUM_PORT)
    return auth


def _call_mcp_tool(tool_name: str, arguments: dict = None) -> dict:
    """汎用 MCP ツール呼び出し（/mcp/execute 経由）"""
    body = {
        **_build_auth(),
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments or {},
        },
    }
    res = requests.post(f"{BRIDGE_URL}/mcp/execute", json=body, timeout=120)
    return res.json()


def call_get_database_list() -> dict:
    """データベース一覧を取得する"""
    return _call_mcp_tool("get_database_list")


def call_get_table_list(database_name: str) -> dict:
    """テーブル一覧を取得する"""
    return _call_mcp_tool("get_table_list", {"databaseName": database_name})


def call_get_table_schema(database_name: str, table_name: str) -> dict:
    """テーブルのスキーマ（列情報）を取得する"""
    return _call_mcp_tool("get_schema", {"databaseName": database_name, "tableName": table_name})


def call_execute_select(sql: str) -> dict:
    """SQLクエリ（SELECT文のみ）を実行してデータを取得する"""
    # LLM がコードブロックで囲む場合があるので除去
    cleaned = sql.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # 最初と最後の ``` 行を除去
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    # SELECT 文のみ許可
    if not cleaned.upper().lstrip().startswith("SELECT"):
        return {
            "success": False,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": f"SELECT文のみ実行可能です。受け取ったSQL: {cleaned[:100]}",
            },
        }
    return _call_mcp_tool("execute_select", {"sql": cleaned})


def call_get_sql_specification() -> dict:
    """Dr.Sum SQL仕様を取得する"""
    return _call_mcp_tool("get_sql_specification")


def call_get_view_definition(database_name: str, view_name: str) -> dict:
    """ビューの定義を取得する"""
    return _call_mcp_tool("get_view_definition", {"databaseName": database_name, "viewName": view_name})


# ── ツール定義（OpenAI Function Calling 用）────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_database_list",
            "description": (
                "Dr.Sum サーバーにあるデータベースの一覧を返します。"
                "「DB一覧を教えて」「どんなデータベースがある？」といった質問に使います。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_table_list",
            "description": (
                "指定したデータベースのテーブル一覧を返します。"
                "「○○DBにはどんなテーブルがある？」「テーブル一覧を教えて」といった質問に使います。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "database_name": {
                        "type": "string",
                        "description": "データベース名（例: 情報系DB）",
                    },
                },
                "required": ["database_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_table_schema",
            "description": (
                "指定したデータベースとテーブルの列名・データ型の一覧を返します。"
                "「○○テーブルの構成を教えて」「カラム一覧を見せて」といった質問に使います。"
                "事前に get_database_list や get_table_list でデータベース名・テーブル名を確認してから使ってください。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "database_name": {
                        "type": "string",
                        "description": "データベース名（例: 情報系DB）",
                    },
                    "table_name": {
                        "type": "string",
                        "description": "テーブル名（例: 売上データ）",
                    },
                },
                "required": ["database_name", "table_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_select",
            "description": (
                "Dr.Sum に対してSELECT SQLクエリを実行し、結果を返します。"
                "「○○テーブルのデータを5件見せて」「○○の売上合計を教えて」といった質問に使います。"
                "Dr.Sum SQL92 準拠の構文で記述してください。"
                "テーブル指定は「データベース名.テーブル名」の形式です。"
                "件数制限には LIMIT を使います。"
                "事前に get_table_schema でカラム構造を確認してからSQLを組み立てることを推奨します。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "実行するSELECT SQLクエリ（例: SELECT * FROM データベース名.テーブル名 LIMIT 10）",
                    },
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sql_specification",
            "description": (
                "Dr.Sum で使用可能な SQL 構文の仕様（関数、句、データ型など）を返します。"
                "「Dr.SumのSQL仕様を教えて」「使える関数は？」といった質問に使います。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_view_definition",
            "description": (
                "指定したデータベースのビュー定義を返します。"
                "「○○ビューの定義を教えて」といった質問に使います。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "database_name": {
                        "type": "string",
                        "description": "データベース名",
                    },
                    "view_name": {
                        "type": "string",
                        "description": "ビュー名",
                    },
                },
                "required": ["database_name", "view_name"],
            },
        },
    },
]

# ── ツール実行ディスパッチ ─────────────────────────────
TOOL_FUNCTIONS = {
    "get_database_list": lambda _args: call_get_database_list(),
    "get_table_list": lambda args: call_get_table_list(args["database_name"]),
    "get_table_schema": lambda args: call_get_table_schema(
        args["database_name"], args["table_name"]
    ),
    "execute_select": lambda args: call_execute_select(args["sql"]),
    "get_sql_specification": lambda _args: call_get_sql_specification(),
    "get_view_definition": lambda args: call_get_view_definition(
        args["database_name"], args["view_name"]
    ),
}


def execute_tool(tool_name: str, arguments: dict) -> str:
    """ツールを実行し、結果をJSON文字列で返す"""
    try:
        func = TOOL_FUNCTIONS.get(tool_name)
        if not func:
            return json.dumps({"error": f"Unknown tool: {tool_name}"}, ensure_ascii=False)

        result = func(arguments)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except requests.exceptions.ConnectionError:
        return json.dumps(
            {"error": "Bridge Server に接続できません。npm start で起動してください。"},
            ensure_ascii=False,
        )
    except requests.exceptions.Timeout:
        return json.dumps(
            {"error": "Bridge Server からの応答がタイムアウトしました。"},
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ── システムプロンプト ─────────────────────────────────
SYSTEM_PROMPT = """\
あなたは Dr.Sum データベースの操作を支援するAIアシスタントです。
ユーザーの自然言語の指示に基づいて、適切なツールを呼び出してください。

## 利用可能ツール
- get_database_list: データベース一覧を取得
- get_table_list: 指定DBのテーブル一覧を取得
- get_table_schema: テーブルのカラム構造を取得
- execute_select: SELECT SQLを実行
- get_sql_specification: SQL仕様を取得
- get_view_definition: ビュー定義を取得

## 重要な制約
- execute_select は **SELECT文のみ** 実行可能です。INSERT, UPDATE, DELETE, CREATE, DROP 等は一切使えません。
- SQLは必ず SELECT で始めてください。コードブロック（```）で囲まないでください。
- SQLは純粋なSQL文のみを渡してください。説明文やコメントを含めないでください。

## ルール
- ユーザーがデータベースやテーブルの名前を知らない場合は、まず get_database_list でDB一覧を取得してください。
- テーブル名がわからない場合は get_table_list でテーブル一覧を確認してください。
- SQLを実行する前に、可能であれば get_table_schema でテーブル構造を確認してください。
- Dr.Sum はSQL92準拠です。テーブルは「データベース名.テーブル名」の形式で指定します。
- 件数制限には LIMIT 句を使います。
- 結果はわかりやすく整形して日本語で回答してください。
- テーブル形式のデータは見やすく表形式で表示してください。
- エラーが発生した場合は、原因と対処方法を説明してください。
"""


# ── メイン会話ループ ───────────────────────────────────
def chat_loop():
    client = create_openai_client()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("=" * 60)
    print("  DrSum 自然言語クライアント")
    print("  自然言語で Dr.Sum を操作できます。")
    print("  終了するには 'exit' または 'quit' と入力してください。")
    print("=" * 60)
    print()

    # Bridge Server 接続確認
    try:
        health = requests.get(f"{BRIDGE_URL}/health", timeout=5)
        if health.status_code == 200:
            print(f"✅ Bridge Server ({BRIDGE_URL}) に接続しました。")
        else:
            print(f"⚠️  Bridge Server ({BRIDGE_URL}) が異常応答を返しました。")
    except requests.exceptions.ConnectionError:
        print(f"⚠️  Bridge Server ({BRIDGE_URL}) に接続できません。")
        print("   Bridge Server を起動してから再度実行してください。")
        print(f"   → cd e:\\MCP\\DrSum-mcp-server\\bridge && npm start")
        return
    print()

    while True:
        # ユーザー入力
        try:
            user_input = input("あなた > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q", "終了"):
            print("終了します。")
            break

        messages.append({"role": "user", "content": user_input})

        # LLM に問い合わせ（ツール呼び出しループ）
        try:
            while True:
                response = client.chat.completions.create(
                    model=AZURE_OPENAI_DEPLOYMENT,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0,
                )
                assistant_message = response.choices[0].message

                # ツール呼び出しがない場合 → 最終応答
                if not assistant_message.tool_calls:
                    reply = assistant_message.content or ""
                    messages.append({"role": "assistant", "content": reply})
                    print(f"\nAI > {reply}\n")
                    break

                # ツール呼び出しがある場合 → 実行して結果を返す
                messages.append(assistant_message)

                for tool_call in assistant_message.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)

                    print(f"  🔧 ツール実行中: {func_name}", end="")
                    if func_args:
                        print(f" ({json.dumps(func_args, ensure_ascii=False)})", end="")
                    print(" ...")

                    result_str = execute_tool(func_name, func_args)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_str,
                    })

                # ループ継続 → LLM がツール結果を受けて最終応答を生成

        except Exception as e:
            error_msg = f"エラーが発生しました: {e}"
            print(f"\n❌ {error_msg}\n")
            # エラーメッセージを会話履歴に追加して続行
            messages.append({"role": "assistant", "content": error_msg})


# ── エントリーポイント ─────────────────────────────────
if __name__ == "__main__":
    if not DRSUM_USER or not DRSUM_PASSWORD:
        print("エラー: DRSUM_USER と DRSUM_PASSWORD を .env に設定してください。")
        sys.exit(1)

    chat_loop()
