"""
DBカタログ(DrSumの辞書テーブル)自然言語チャットクライアント
    DB-Catalog_chat11.py システムプロンプトを外部ファイルにしたバージョン + 配布用
    DB-Catalog_chat12.py トークン量を表示するバージョン
基本機能:自然言語でDBカタログの探索を指示すると、辞書テーブルを参照して探索結果を回答する。
追加予定機能:１．カラムの値（記号）を名称に変換する。２．テーブルを作成したクエリーを回答する。
技術概要:Azure OpenAI の Function Calling を使い、DrSum MCP Bridge Server の
        API 呼び出しに変換して実行する。
使い方:
  1. .env ファイルを作成して環境変数を設定する。
  2. pip install -r requirements.txt を実行してライブラリをインストールする。
  3. python drsum_chat.py を実行する。
  注) 仮想フォルダ .venvを作成するのが望ましい。
  システムプロンプトを変更する場合は、system_prompt.txtとは別のファイル（例：`prompt_analysis.txt`）を作成し、
  そこにプロンプトを記述してから、.envのSYSTEM_PROMPT_FILEにファイル名を指定してください。
"""

import json
import os
import sys

import requests
from dotenv import load_dotenv
from openai import AzureOpenAI

# ── 環境変数読み込み ──────────────────────────────────
# 実行ファイルまたはスクリプトのディレクトリを取得
if getattr(sys, 'frozen', False):
    # .exe化されている場合
    base_dir = os.path.dirname(sys.executable)
else:
    # 通常のPython実行の場合
    base_dir = os.path.dirname(os.path.abspath(__file__))

# .envのフルパスを指定して読み込む
env_path = os.path.join(base_dir, ".env")
load_dotenv(env_path)

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


def call_execute_select(database_name: str, select_statement: str, limit: int = None) -> dict:
    """SELECT文を実行してデータを取得する"""
    # LLM がコードブロックで囲む場合があるので除去
    cleaned = select_statement.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
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

    args = {
        "databaseName": database_name,
        "selectStatement": cleaned,
    }
    if limit is not None:
        args["limit"] = limit

    return _call_mcp_tool("execute_select", args)


def call_get_sql_specification(index: int = None) -> dict:
    """Dr.Sum SQL仕様を取得する"""
    args = {}
    if index is not None:
        args["index"] = index
    return _call_mcp_tool("get_sql_specification", args)


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
                "指定したデータベースのテーブル・ビュー一覧を返します。"
                "Type: 0=Table, 1=View, 2=Multiview, 3=Link table。"
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
                "Dr.Sum に対してSELECTクエリを実行し、結果を返します。"
                "SELECT文のみ実行可能です（INSERT/UPDATE/DELETE等は不可）。"
                "「○○テーブルのデータを見せて」「○○の合計を教えて」といった質問に使います。"
                "重要: database_name にデータベース名を指定し、select_statement にはデータベース名を含めずテーブル名だけで記述してください。"
                "例: database_name='情報系DB', select_statement='SELECT * FROM VWSDT0100 LIMIT 5'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "database_name": {
                        "type": "string",
                        "description": "クエリ実行対象のデータベース名（例: 情報系DB）",
                    },
                    "select_statement": {
                        "type": "string",
                        "description": "実行するSELECT文。データベース名プレフィックスは付けず、テーブル名だけで記述する（例: SELECT * FROM VWSDT0100 LIMIT 5）",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "最大取得件数（デフォルト: 50、最大: 2000）。SELECT文内のLIMIT句の代わりにこちらを使用しても可。",
                    },
                },
                "required": ["database_name", "select_statement"],
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
                "indexを指定すると特定の仕様を取得し、指定しないと目次を返します。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "取得する仕様のインデックス番号。省略すると目次を返す。",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_view_definition",
            "description": (
                "指定したビューのSQL定義を返します。テーブルやリンクテーブルには使えません。"
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
    "execute_select": lambda args: call_execute_select(
        args["database_name"], args["select_statement"], args.get("limit")
    ),
    "get_sql_specification": lambda args: call_get_sql_specification(args.get("index")),
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
def load_system_prompt(file_path="system_prompt.txt"):
    """外部ファイルからシステムプロンプトを読み込む"""
    try:
        # UTF-8で読み込み（日本語が含まれるため）
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"警告: {file_path} が見つかりません。デフォルトのプロンプトを使用します。")
        return "あなたはDr.Sumのアシスタントです。"
    except Exception as e:
        print(f"エラー: プロンプトの読み込み中に問題が発生しました: {e}")
        sys.exit(1)

SYSTEM_PROMPT_FILE = os.getenv("SYSTEM_PROMPT_FILE", "system_prompt.txt")

# ── メイン会話ループ ───────────────────────────────────
def chat_loop():
    client = create_openai_client()
    # SYSTEM_PROMPTの読込＆設定
    system_content = load_system_prompt(SYSTEM_PROMPT_FILE)
    messages = [{"role": "system", "content": system_content}]

    # ── 累計トークン管理用 ──
    total_prompt_tokens = 0
    total_completion_tokens = 0

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

                # --- トークン使用量の加算 ---
                usage = response.usage
                total_prompt_tokens += usage.prompt_tokens
                total_completion_tokens += usage.completion_tokens
                print(f"  [今回: Prompt {usage.prompt_tokens} / Comp {usage.completion_tokens} tokens]")


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

                    result_str = execute_tool(tool_call.function.name, json.loads(tool_call.function.arguments))
                

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_str,
                    })

                # ループ継続 → LLM がツール結果を受けて最終応答を生成

        except Exception as e:
            error_msg = f"エラーが発生しました: {e}"
            print(f"\n❌ {error_msg}\n")
            messages.append({"role": "assistant", "content": error_msg})

        # ── 終了時の集計表示とポーズ ──
        print("\n" + "=" * 60)
        print("  チャットセッション統計")
        print(f"  ・総入力トークン: {total_prompt_tokens}")
        print(f"  ・総出力トークン: {total_completion_tokens}")
        print(f"  ・合計トークン  : {total_prompt_tokens + total_completion_tokens}")
        print("=" * 60)
    
        input("\nEnterキーを押すと画面を閉じます...") # ここで一時停止します

# ── エントリーポイント ─────────────────────────────────
if __name__ == "__main__":
    if not DRSUM_USER or not DRSUM_PASSWORD:
        print("エラー: DRSUM_USER と DRSUM_PASSWORD を .env に設定してください。")
        sys.exit(1)

    chat_loop()
