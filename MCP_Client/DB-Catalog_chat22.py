"""
DBカタログ(DrSumの辞書テーブル)自然言語チャットクライアント
    DB-Catalog_chat11.py システムプロンプトを外部ファイルにしたバージョン + 配布用
    DB-Catalog_chat12.py トークン量を表示するバージョン & 終了後に総トークンを表示してから画面を閉じる
    DB-Catalog_chat13.py TOOLSを外部読込にしたバージョン（たぶん、間違っているのでは？）
    DB-Catalog_chat20.py TOOLS定義、システムプロンプト、call_mcp_toolの動作を変更し、DBカタログ用にしたバージョン
    　また、TOOLS_CONFIG.jsonファイルも.env20を変更すると自動で変更される（コード修正不要）となっている

基本機能:自然言語でDBカタログの探索を指示すると、辞書テーブルを参照して探索結果を回答する。
追加予定機能:１．カラムの値（記号）を名称に変換する。２．テーブルを作成したクエリーを回答する。
技術概要:Azure OpenAI の Function Calling を使い、DrSum MCP Bridge Server の
        API 呼び出しに変換して実行する。
使い方:
  1. .env20 ファイルを作成して環境変数を設定する。
  2. pip install -r requirements.txt を実行してライブラリをインストールする。
  3. python drsum_chat.py を実行する。
  注) 仮想フォルダ .venvを作成するのが望ましい。
  システムプロンプトを変更する場合は、system_prompt.txtとは別のファイル（例：`prompt_analysis.txt`）を作成し、
  そこにプロンプトを記述してから、.env20のSYSTEM_PROMPT_FILEにファイル名を指定してください。
"""

import json
import os
from pyexpat.errors import messages
import sys
from unittest import result

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

# .env20のフルパスを指定して読み込む
env_path = os.path.join(base_dir, ".env20")
load_dotenv(env_path)

# 1. 環境変数からファイル名を取得（なければデフォルト tools_config.json）
TOOLS_CONFIG_FILE = os.getenv("TOOLS_CONFIG_FILE", "tools_config.json")

# 2. 関数定義（ここは今のままでもOKですが、引数で受け取るようにします）
def load_tools_config(file_path):
    """外部JSONからツール定義を読み込む"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"ツール定義の読み込みエラー ({file_path}): {e}")
        sys.exit(1)


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
        print("エラー: AZURE_OPENAI_ENDPOINT と AZURE_OPENAI_API_KEY を .env20 に設定してください。")
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

# トークン削減のために追加　2026/3/13
def compact_execute_select_text(result: dict, max_chars: int = 2500, head_lines: int = 25) -> dict:
    """
    execute_select の戻りが content[0].text (文字列) の場合に、
    LLMに渡す情報量を落としすぎずに縮約する。
    """
    try:
        r = (result or {}).get("result") or {}
        content = r.get("content") or []
        is_error = bool(r.get("isError"))

        text = ""
        if isinstance(content, list) and content and isinstance(content[0], dict):
            text = content[0].get("text") or ""
        else:
            # 想定外でも落とさない
            text = str(content)[:max_chars]

        # 行単位で先頭だけ残す（表形式のときに効く）
        lines = text.splitlines()
        head = "\n".join(lines[:head_lines]).strip()

        # 文字数がまだ長い場合は先頭/末尾を残して省略
        if len(head) > max_chars:
            head = head[:max_chars] + "\n…(truncated)…"

        compact = {
            "success": result.get("success", True),
            "meta": {
                "isError": is_error,
                "requestId": result.get("requestId"),
                "duration": result.get("duration"),
                "original_lines": len(lines),
                "kept_lines": min(len(lines), head_lines),
                "truncated": len(lines) > head_lines or len(text) > len(head),
            },
            "text_head": head
        }
        return compact
    except Exception as e:
        # デバッグしやすい形で返す
        return {"success": False, "error": f"compact_execute_select_text failed: {e}"}


def execute_tool(tool_name: str, arguments: dict) -> str:
    """ツールを実行し、結果をJSON文字列で返す"""
    try:
        func = TOOL_FUNCTIONS.get(tool_name)
        if not func:
            return json.dumps({"error": f"Unknown tool: {tool_name}"}, ensure_ascii=False, separators=(",", ":"))

        result = func(arguments)

        # ★ execute_select の戻りは type/text なので縮約して渡す
        if tool_name == "execute_select" and isinstance(result, dict):
            result = compact_execute_select_text(result, max_chars=2500, head_lines=25)

        # ★ JSONはインデント無し（空白削減）
        return json.dumps(result, ensure_ascii=False, separators=(",", ":"))

    except requests.exceptions.ConnectionError:
        return json.dumps({"error": "Bridge Server に接続できません。npm start で起動してください。"},
                          ensure_ascii=False, separators=(",", ":"))
    except requests.exceptions.Timeout:
        return json.dumps({"error": "Bridge Server からの応答がタイムアウトしました。"},
                          ensure_ascii=False, separators=(",", ":"))
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False, separators=(",", ":"))


# ── システムプロンプト ─────────────────────────────────
# 1. まず環境変数を読み込んでおく
SYSTEM_PROMPT_FILE = os.getenv("SYSTEM_PROMPT_FILE", "system_prompt.txt")

# 2. 関数側で「指定がなければ環境変数の値を使う」ようにする
def load_system_prompt(file_path=None):
    # もし引数(file_path)が空なら、環境変数の値を入れる
    if file_path is None:
        file_path = SYSTEM_PROMPT_FILE
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"警告: {file_path} が見つかりません。")
        return "あなたはDr.Sumのアシスタントです。"
    except Exception as e:
        print(f"エラー: プロンプトの読み込み中に問題が発生しました: {e}")
        sys.exit(1)

# プログラム実行時にTOOLS読み込み
TOOLS = load_tools_config(TOOLS_CONFIG_FILE)
# トークン削減のため、execute_select以外は除外する
TOOLS = [t for t in TOOLS if t.get("function", {}).get("name") == "execute_select"]

# ── トークン削減のためのメッセージ整理関数─────────────────────────
def keep_last_tool_only(messages, keep=1):
    """
    messages には dict だけでなく ChatCompletionMessage 等のオブジェクトが混在する。
    そのため role 取得は dict.get ではなく、dict/obj 両対応で行う。
    """
    system = messages[:1]
    rest = messages[1:]

    def get_role(m):
        if isinstance(m, dict):
            return m.get("role")
        # ChatCompletionMessage など（属性で role を持つ）
        return getattr(m, "role", None)

    tool_pos = [i for i, m in enumerate(rest) if get_role(m) == "tool"]
    drop = set(tool_pos[:-keep]) if len(tool_pos) > keep else set()
    rest2 = [m for i, m in enumerate(rest) if i not in drop]
    return system + rest2

# ── メイン会話ループ ───────────────────────────────────
def chat_loop():
    client = create_openai_client()
    # SYSTEM_PROMPTの読込
    system_content = load_system_prompt(SYSTEM_PROMPT_FILE)
    messages = [{"role": "system", "content": system_content}]
    
    # 【重要】ループの外側で必ず 0 を代入して初期化する
    total_prompt = 0
    total_completion = 0

    print("=" * 60)
    print("  DrSum 自然言語クライアント")
    print("  終了するには 'exit' または 'quit' と入力してください。")
    print("=" * 60)
    print()

    # Bridge Server 接続確認 (中略)
    try:
        health = requests.get(f"{BRIDGE_URL}/health", timeout=5)
        if health.status_code == 200:
            print(f"✅ Bridge Server ({BRIDGE_URL}) に接続しました。")
    except:
        print(f"⚠️ Bridge Server に接続できません。")

    # メインループ開始
    try:
        while True:
            try:
                user_input = input("あなた > ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q", "終了"):
                break

            messages.append({"role": "user", "content": user_input})

            # LLM 問い合わせ（ツール呼び出しループ）
            while True:
                response = client.chat.completions.create(
                    model=AZURE_OPENAI_DEPLOYMENT,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0,
                )
                
                # --- トークン情報の加算 ---
                usage = response.usage
                total_prompt += usage.prompt_tokens
                total_completion += usage.completion_tokens
                print(f"  [Tokens: {usage.prompt_tokens} In / {usage.completion_tokens} Out]")

                assistant_message = response.choices[0].message

                # ツール呼び出しがない場合 → 最終応答
                if not assistant_message.tool_calls:
                    reply = assistant_message.content or ""
                    messages.append({"role": "assistant", "content": reply})
                    print(f"\nAI > {reply}\n")
                    break

                # ツール呼び出しがある場合
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
                    
                    # ★ 古いtool結果は捨てる（直近1つだけ）
                    messages = keep_last_tool_only(messages, keep=1)

    except Exception as e:
        # 途中でエラーが起きても、ここを通ることでトークン数を表示して止まる
        print(f"\n❌ 実行中にエラーが発生しました: {e}")

    # ── 会話ループ終了後の処理 ──
    # ここは while True の外側なので、exit時やエラー時に一度だけ実行されます
    print("\n" + "=" * 60)
    print("  チャットセッション統計")
    print(f"  ・総入力トークン: {total_prompt}")
    print(f"  ・総出力トークン: {total_completion}")
    print(f"  ・合計トークン  : {total_prompt + total_completion}")
    print("=" * 60)
    
    input("\nEnterキーを押すと画面を閉じます...")
     
# ── エントリーポイント ─────────────────────────────────
if __name__ == "__main__":
    if not DRSUM_USER or not DRSUM_PASSWORD:
        print("エラー: DRSUM_USER と DRSUM_PASSWORD を .env20 に設定してください。")
        sys.exit(1)

    chat_loop()
