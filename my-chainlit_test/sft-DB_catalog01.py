# chainlit run sft-DB_catalog01.py -w
# chainlit + AutoGen(Azure OpenAI) + MCP を元にDBカタログ改造
# 接続が切れてしまうコードになっていたので、Copilotに手直ししてもらった
# それを元に辞書を参照するように改造したが、エラーが多く出て沢山改造した。
# ところが、参照していたのが本番サーバで、辞書がない！　⇒　動いたが改造しすぎたかも。
import asyncio
import inspect
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

        # tool schema は inject 用に保持（runner内で更新）
        self.tool_schema_map: dict[str, dict] = {}

        # runner連携
        self._queue: asyncio.Queue = asyncio.Queue()
        self._runner_task: asyncio.Task | None = None
        self._ready = asyncio.Event()
        self._closed = asyncio.Event()

    # 互換のため open() も残す（呼び出し側を触らない場合）
    async def open(self, server_params: StdioServerParameters):
        await self.start(server_params)

    async def start(self, server_params: StdioServerParameters):
        """MCP を専用タスクで起動（多重起動しない）"""
        if self._runner_task and not self._runner_task.done():
            return
        self._ready.clear()
        self._closed.clear()
        self._runner_task = asyncio.create_task(self._runner(server_params))
        await self._ready.wait()

    async def close(self):
        """runner に終了要求を出し、閉じる（exitはrunner内で実行される）"""
        if not self._runner_task:
            return
        await self._queue.put(None)
        await self._closed.wait()
        await self._runner_task

    async def _runner(self, server_params: StdioServerParameters):
        """async with の enter/exit をこのタスク内に閉じ込める"""
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._ready.set()

                    while True:
                        item = await self._queue.get()
                        if item is None:
                            break

                        op, payload, fut = item
                        try:
                            if op == "list_tools":
                                tools = await session.list_tools()

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

                                # ★元の形式 {"tools":[...]} を維持
                                fut.set_result({"tools": tools_out})

                            elif op == "call_tool":
                                name, arguments = payload
                                result = await session.call_tool(name, arguments)
                                fut.set_result(_mcp_result_to_plain_dict(result))
                            else:
                                fut.set_result({"isError": True, "content": [f"Unknown op: {op}"]})

                        except Exception as e:
                            fut.set_result({"isError": True, "content": [f"{type(e).__name__}: {e}"]})

        finally:
            self._closed.set()
            self._ready.set()  # start待ちが残らないよう保険

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
        fut = asyncio.get_running_loop().create_future()
        await self._queue.put(("list_tools", None, fut))
        payload = await fut
        return json.dumps(payload, ensure_ascii=False, indent=2)

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

        # sql互換
        if name == "execute_select" and isinstance(arguments, dict):
            if "selectStatement" not in arguments and "sql" in arguments:
                arguments["selectStatement"] = arguments.pop("sql")

        # inject のため初回は tools を取得しておく（tool_schema_map を埋める）
        if not self.tool_schema_map:
            await self.list_tools_json()

        arguments = self._inject_auth_if_needed(name, arguments)

        fut = asyncio.get_running_loop().create_future()
        await self._queue.put(("call_tool", (name, arguments), fut))
        payload = await fut
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

    import inspect
    print("=== MCPContext debug ===")
    print("MCPContext file:", inspect.getsourcefile(MCPContext))
    print("MCPContext line:", inspect.getsourcelines(MCPContext)[1])
    print("list_tools_json file:", inspect.getsourcefile(mcp_ctx.list_tools_json))
    print("list_tools_json line:", mcp_ctx.list_tools_json.__code__.co_firstlineno)
    print("has lock attr?:", hasattr(mcp_ctx, "lock"))
    print("dict keys:", list(mcp_ctx.__dict__.keys()))
    print("========================")


    await mcp_ctx.start(server_params)

    # tool関数は mcp_ctx を参照する
    agent_tools = [
        FunctionTool(mcp_ctx.list_tools_json, name="mcp_list_tools", description="MCPサーバーが提供するtools一覧を取得する"),
        FunctionTool(mcp_ctx.call_tool_json, name="mcp_call_tool", description="指定したMCP toolを呼び出す。arguments(dict)推奨。互換として arguments_json(文字列JSON) も可。"),
    ]

    ROLE_INSTRUCTIONS = """
あなたは「Dr.Sum MCP クライアント専任アシスタント（辞書テーブル限定）」です。
目的：Dr.Sum のローカル MCP サーバー経由で、
辞書DB（カタログ） x00_META_DB 内の以下2テーブルのみを対象に、
メタデータ把握と少量データ確認を行い、日本語で分かりやすく報告します。

  - テーブル辞書：dd_tables
  - カラム辞書： dd_columns

【対象範囲の厳格な制約（最重要）】
- 対象DB（catalog）は **必ず x00_META_DB**。
- 参照してよいテーブルは **dd_tables と dd_columns のみ**。
- 上記以外の DB / テーブル / ビュー / マルチビュー / リンクテーブルを
  調査・参照・推測・提案してはいけない。

【ツール利用の絶対ルール】
- MCP ツール呼び出しは必ず FunctionTool『mcp_call_tool』経由で行う。
- 以下のツールでは **必ず databaseName="x00_META_DB" を指定**する：
  - get_table_list
  - get_schema
  - execute_select
  - measure_select
  - get_view_definition（※ただし本ロールでは原則使用しない）
- 『mcp_list_tools』は次の場合に限り実行可：
  1) 初回実行時
  2) 不明なツール名・引数が必要になったとき
  3) inputSchema を再確認する必要があるとき
  4) ツールエラーが続き、原因切り分けが必要なとき

【標準手順（辞書2表・DB完全固定）】
1) get_database_list を実行し、x00_META_DB の存在を確認
2) 次の2テーブルのみを対象として処理を行う（列挙は原則しない）：
   - dd_tables
   - dd_columns
3) 各テーブルについて必ず以下を実行：
   (a) get_schema（databaseName="x00_META_DB", tableName=対象名）
   (b) execute_select で先頭5行を取得（limit=5）
4) 必要に応じて、辞書2表の範囲内で最小限の追加 SELECT を実行
   - 例：dd_columns を特定テーブル名で絞る等
   - limit は小さく保つ

【例外（最小限の列挙を許可する条件）】
- dd_tables / dd_columns に対する get_schema や execute_select が
  「存在しない」「権限不足」等で失敗した場合のみ、
  get_table_list（databaseName="x00_META_DB"）を実行して存在確認を行ってよい。
- 列挙結果から参照してよいのは、dd_tables / dd_columns と
  明確に一致すると判断できるもののみ。

【SQL（execute_select）方針：安全第一】
- SQL は SELECT のみ。INSERT / UPDATE / DELETE 等は禁止。
- FROM 句に指定してよいのは **dd_tables または dd_columns のみ**。
- databaseName はツール引数で固定するため、
  SQL では DB 名を付けない：
    例）SELECT * FROM dd_tables
- 識別子は原則クォート無しで記述する。
- SQL 側で DB 名（x00_META_DB）を指定しようとしてはいけない。
- limit は基本 5。必要な場合でも最大 50 まで。
- カラム名は get_schema で確認したもののみを使用する。

【Dr.Sum スキーマ type コード定義（get_schema columns[].type）】
- type=0  : VARCHAR / CHAR（文字列型）
- type=1  : INTEGER（整数型）
- type=2  : REAL（浮動小数型）
- type=3  : DATE（日付型）
- type=4  : TIME（時刻型）
- type=5  : TIMESTAMP（日時型）
- type=7  : NUMERIC（数値型）
- type=12 : INTERVAL（期間型）
※ 上記以外は「不明（type=<数値>）」として扱う。

【型表記ルール】
- 出力のデータ型は以下の形式で表記する：
  日本語分類（Native Type / type=<数値>）
- type 値からの推測は禁止。対応表に厳密に従う。

【出力形式（辞書テーブル向け）】
- 「概要」
- 「dd_tables 確認結果（スキーマ／先頭データ）」
- 「dd_columns 確認結果（スキーマ／先頭データ）」
- 「辞書構造から読み取れる示唆」
- 「次アクション提案（辞書2表の範囲内）」

- 実行したツール名と引数を必ず箇条書きで併記（再現性確保）。
- 取得データは先頭行を簡易テーブル形式で表示。
- エラー時は、DB名（x00_META_DB）、テーブル名、
  権限・接続の確認ポイントを明記する。

【注意】
- SELECT 以外のSQLは禁止。
- 機密情報を出力に含めない。
- 辞書2表の範囲を超える分析・推測・提案をしない。
- get_schema / execute_select などの “MCPサーバー側のツール名” は、
  直接呼び出せない。必ず mcp_call_tool を使い、
  { "name": "<tool名>", "arguments": {...} } の形で呼び出すこと。
- databaseName は必ず "x00_META_DB"。
  例：mcp_call_tool(name="get_schema", arguments={"databaseName":"x00_META_DB","tableName":"dd_tables"})  
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