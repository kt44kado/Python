# chainlit + AutoGen(Azure OpenAI) + MCP のサンプル
# chainlit run sft-chainlit03.py -w
# 接続が切れてしまうコードになっていたので、Copilotに手直ししてもらった
import asyncio
import json
import os
import chainlit as cl
from dotenv import load_dotenv
import truststore

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core.tools import FunctionTool

from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

truststore.inject_into_ssl()
load_dotenv()

def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"環境変数 {name} が未設定です。.env を確認してください。")
    return v

def _optional_env(name: str) -> str | None:
    v = os.getenv(name)
    return v if v else None

def _try_parse_json(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x

def _mcp_result_to_plain_dict(result) -> dict:
    content_out = []
    for c in (result.content or []):
        if hasattr(c, "text"):
            content_out.append(_try_parse_json(c.text))
        elif hasattr(c, "model_dump"):
            content_out.append(c.model_dump())
        else:
            content_out.append(_try_parse_json(str(c)))
    return {"isError": bool(getattr(result, "isError", False)), "content": content_out}

def _schema_properties(schema: dict) -> set[str]:
    props = schema.get("properties", {}) if isinstance(schema, dict) else {}
    return set(props.keys())

class MCPContext:
    def __init__(self, drsum_user, drsum_password, drsum_host, drsum_port):
        self.drsum_user = drsum_user
        self.drsum_password = drsum_password
        self.drsum_host = drsum_host
        self.drsum_port = drsum_port

        self.tool_schema_map: dict[str, dict] = {}
        self.lock = asyncio.Lock()

        self.mcp_cm = None          # stdio_client context manager
        self.session_cm = None      # ClientSession context manager
        self.mcp_session = None     # ClientSession instance
        self.read = None
        self.write = None

    async def open(self, server_params: StdioServerParameters):
        # async with を使わず手動で enter して保持する
        self.mcp_cm = stdio_client(server_params)
        self.read, self.write = await self.mcp_cm.__aenter__()

        self.session_cm = ClientSession(self.read, self.write)
        self.mcp_session = await self.session_cm.__aenter__()

        await self.mcp_session.initialize()

    async def close(self):
        # 逆順で確実に閉じる
        if self.session_cm:
            await self.session_cm.__aexit__(None, None, None)
        if self.mcp_cm:
            await self.mcp_cm.__aexit__(None, None, None)

    def _inject_auth_if_needed(self, tool_name: str, args: dict) -> dict:
        schema = self.tool_schema_map.get(tool_name) or {}
        props = _schema_properties(schema)

        candidates_user = ["user", "userid", "userId", "username", "loginId", "login_id"]
        candidates_pass = ["password", "pass", "pwd"]

        if any(k in props for k in candidates_user):
            for k in candidates_user:
                if k in props and k not in args:
                    args[k] = self.drsum_user
                    break

        if any(k in props for k in candidates_pass):
            for k in candidates_pass:
                if k in props and k not in args:
                    args[k] = self.drsum_password
                    break

        if self.drsum_host:
            for k in ["host", "server", "hostname", "endpoint"]:
                if k in props and k not in args:
                    args[k] = self.drsum_host
                    break

        if self.drsum_port:
            for k in ["port"]:
                if k in props and k not in args:
                    try:
                        args[k] = int(self.drsum_port)
                    except ValueError:
                        args[k] = self.drsum_port
                    break

        return args

    async def list_tools_json(self) -> str:
        async with self.lock:
            tools = await self.mcp_session.list_tools()

        self.tool_schema_map.clear()

        tools_out = []
        for t in tools.tools:
            schema = t.inputSchema or {}
            self.tool_schema_map[t.name] = schema

            props = list((schema.get("properties") or {}).keys()) if isinstance(schema, dict) else []
            required = schema.get("required", []) if isinstance(schema, dict) else []
            tools_out.append({
                "name": t.name,
                "description": t.description,
                "properties": props,
                "required": required,
            })

        return json.dumps({"tools": tools_out}, ensure_ascii=False, indent=2)

    async def call_tool_json(self, name: str, arguments: dict | None = None, arguments_json: str | None = None) -> str:
        if arguments is None:
            if arguments_json:
                try:
                    arguments = json.loads(arguments_json)
                except json.JSONDecodeError as e:
                    return json.dumps(
                        {"isError": True, "content": [f"arguments_json が不正な JSON です: {e}"]},
                        ensure_ascii=False, indent=2
                    )
            else:
                arguments = {}

        if name == "execute_select" and isinstance(arguments, dict):
            if "selectStatement" not in arguments and "sql" in arguments:
                arguments["selectStatement"] = arguments.pop("sql")

        arguments = self._inject_auth_if_needed(name, arguments)

        async with self.lock:
            result = await self.mcp_session.call_tool(name, arguments)

        payload = _mcp_result_to_plain_dict(result)
        return json.dumps(payload, ensure_ascii=False, indent=2)


@cl.on_chat_start
async def setup_agent():
    model_client = AzureOpenAIChatCompletionClient(
        api_key=os.getenv("API_KEY"),
        azure_endpoint=os.getenv("API_ENDPOINT"),
        api_version=os.getenv("API_VERSION"),
        azure_deployment=os.getenv("DEPLOYMENT_NAME"),
        model=os.getenv("DEFAULT_MODEL_NAME"),
        temperature=0.2,
    )

    drsum_user = _require_env("DRSUM_USER")
    drsum_password = _require_env("DRSUM_PASSWORD")
    drsum_host = _optional_env("DRSUM_HOST")
    drsum_port = _optional_env("DRSUM_PORT")

    args = [
        "-Dfile.encoding=UTF-8",
        "-jar",
        r"C:\drsum-mcp-server\drsum-local-mcp-server-1.1.00.0000.jar",
        f"--user={drsum_user}",
        f"--password={drsum_password}",
    ]
    if drsum_host:
        args.append(f"--host={drsum_host}")
    if drsum_port:
        args.append(f"--port={drsum_port}")

    server_params = StdioServerParameters(
        command="java",
        args=args,
        env=None,
    )

    # ★MCP接続を開いて保持
    mcp_ctx = MCPContext(drsum_user, drsum_password, drsum_host, drsum_port)
    await mcp_ctx.open(server_params)

    # tool関数は mcp_ctx を参照する
    agent_tools = [
        FunctionTool(mcp_ctx.list_tools_json, name="mcp_list_tools", description="MCPサーバーが提供するtools一覧を取得する"),
        FunctionTool(mcp_ctx.call_tool_json, name="mcp_call_tool", description="指定したMCP toolを呼び出す。arguments(dict)推奨。互換として arguments_json(文字列JSON) も可。"),
    ]

    ROLE_INSTRUCTIONS = """
あなたは「Dr.Sum MCP クライアント専任アシスタント」です。
目的：Dr.Sum のローカル MCP サーバー経由で、メタデータの把握と少量データの確認を行い、結果を日本語で分かりやすく報告します。

【ツール利用ルール】
- MCPのツール呼び出しは必ず FunctionTool『mcp_call_tool』経由で行うこと。
- 『mcp_list_tools』は以下の場合に限り実行してよい（毎回必須ではない）：
  1) 初回実行時（ツール名や引数が不明なとき）
  2) 不明なツール名が必要になったとき
  3) inputSchemaを確認しないと引数が確定できないとき
  4) ツール呼び出しエラーが続き、原因切り分けが必要なとき
- 認証／接続確立が必要そうなツールが存在する場合は、最初にそれを実行すること。
  - user/password/host/port はクライアントから自動注入される（inputSchema にキーがある場合）。

【標準手順】
1) DB（またはカタログ／スキーマ）一覧を取得（get_database_list）
2) 1つ選んでテーブル一覧（tableType=0）とビュー一覧（tableType=1）を取得（get_table_list）
3) 代表として先頭のテーブル（またはビュー）について、必ず以下を実行：
   (a) get_schema で列定義を取得
   (b) execute_select で先頭5行（limit=5）を取得
4) 必要に応じて get_view_definition で補足情報を取得

【SQL（execute_select）方針】
- execute_select のSQL文字列は、必ず引数名 selectStatement で渡す（sql というキーは使わない）。
- SELECTは原則「クォート無し」で試す（例: SELECT * FROM テーブル名）。
- 解析エラー等で失敗した場合のみ、識別子クォートを試す（例: SELECT * FROM "テーブル名"）。
- limit は基本 5、必要な場合のみ段階的に増やす。

【Dr.Sum スキーマ type コード定義（get_schema の columns[].type）】
- type=0  : VARCHAR / CHAR（文字列型）
- type=2  : NUMERIC / INTEGER（数値型）
- type=3  : REAL（浮動小数型）
- type=4  : DATE（日付型）
- type=5  : TIME（時刻型）
- type=6  : TIMESTAMP（日時型）
- type=12 : INTERVAL（期間型）

【型表記ルール】
- 出力の「データ型」は次の形式で表示する： 日本語分類（Native Type / type=<数値>）
  例）文字列型（VARCHAR / type=0）
- type 値からデータ型を “推測” してはいけない。必ず上記の対応表に従う。
- 未定義の type は「不明（type=<数値>）」と出力し、勝手に決めない。

【出力形式】
- 「概要」→「詳細結果」→「次アクション提案」の順で日本語で整理。
- 実行したツール名と引数を箇条書きで併記（再現性のため）。
- 取得データの先頭行はテーブル風に整形（可能な範囲で）。
- エラーや空結果時は、認証／権限／対象名／接続先などの確認ポイントを明記。

【注意】
- SELECT 以外のSQLは実行しない（execute_select は読み取り専用）。
- 機密情報（パスワード等）を出力に含めない。
"""

    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        tools=agent_tools,
        system_message=ROLE_INSTRUCTIONS,
        reflect_on_tool_use=True,
        max_tool_iterations=8,
    )

    cl.user_session.set("agent", agent)
    cl.user_session.set("mcp_ctx", mcp_ctx)


@cl.on_chat_end
async def teardown():
    # ★終了時にクリーンアップ
    mcp_ctx = cl.user_session.get("mcp_ctx")
    if mcp_ctx:
        await mcp_ctx.close()


@cl.on_message
async def run_conversation(message: cl.Message):
    agent = cl.user_session.get("agent")
    result = await agent.run(task=message.content)
    await cl.Message(content=result.messages[-1].content, author="Assistant").send()