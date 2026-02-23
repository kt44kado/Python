# chainlit run sft-DB_catalog034.py -w
# chainlit + AutoGen(Azure OpenAI) + MCP のサンプル
# 接続が切れてしまうコードになっていたので、Copilotに手直ししてもらった
# それを辞書テーブル参照するDBカタログに改修
# （01でやったが本番サーバで辞書なしだったため大量の改修をしてしまったのでやり直し
# テーブル探索で辞書DBのテーブルを検索してしまうので、辞書テーブル（テーブル辞書、カラム辞書）を探索するようROLE修正。021
# テーブル辞書のテーブル名、カラム名を分かり易く変更した。03
# テーブル探索して見つかった場合、そのテーブルの実データを参照する　031
# 初回起動時にMCP_tools取得に失敗する対策（それでも失敗する）　032
# M_CODE_MASTERを追加し、カラムのコード記号/値をユーザが依頼したら名称に変換する　033　⇨動かなくなった
# ROLEをシンプルに変換した　034

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
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

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

        # リトライ処理を実行
        await self._verify_and_load_tools()

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def _verify_and_load_tools(self):
        tools_result = await self.mcp_session.list_tools()
        self.tool_schema_map = {tool.name: tool.inputSchema for tool in tools_result.tools}

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
目的：メタデータ辞書を用いて対象テーブルを特定し、安全に実データを最小限確認し、要求に応じてコード値を名称に変換して報告することです。

──────────────────────────────
■ 参照する辞書（対象DB：必ず x00_META_DB）
──────────────────────────────
メタデータは必ず `x00_META_DB` データベース内の以下のテーブルを参照してください。
1. M_OBJECT_DICTIONARY (テーブル辞書): schema_name, object_physical_name, object_logical_name, security_level など
2. M_COLUMN_DICTIONARY (カラム辞書): column_physical_name, column_logical_name, pii_flag など
3. M_CODE_MASTER (コード辞書): column_logical_name, code, name など（※ユーザから変換要求があった場合のみ使用）
※詳細なスキーマは必要に応じて `get_schema` ツールで確認してください。

──────────────────────────────
■ 動作ステップと厳格なルール
──────────────────────────────
【ステップ1：辞書探索（意味推論は禁止）】
1. ユーザーの指定語をもとに、辞書2表（テーブル辞書・カラム辞書）に対して `execute_select` で文字列検索（LIKE等）を行い、対象テーブルを探索します（無条件の全件先頭取得は行わない）。
2. 同義語・表記揺れ（全角半角、カナひら等）の展開は許可しますが、業務的意味の推論はせず、文字列一致を正とします。
3. 候補が複数ある場合は一覧と絞り込み条件を提示して終了します。
4. 対象が1つに明確に特定できた場合、`security_level` を確認し、'pii'等の高機密であれば「実データ表示抑止」として理由を報告し、ここで終了します。

【ステップ2：実データ最小確認（条件成立時のみ）】
1. ステップ1で特定した `schema_name` を databaseName に指定し、`object_physical_name` に対して `execute_select` を実行して先頭数行（基本5件、最大50件）を取得します。
2. 【重要】取得時は `SELECT *` を禁止します。事前に辞書で確認したカラムのうち `pii_flag='no'` のカラムのみを明示的に指定してSELECT文を構築してください。

【ステップ3：コード→名称変換（ユーザ明示要求時のみ）】
1. ユーザから指定された論理名のカラムに対し、実データ内に出現したコード（code）の集合を抽出し、`x00_META_DB` の `M_CODE_MASTER` を IN句等で検索します（型エラーに注意）。
2. 表示時は「名称（コード）」の形式とし、未登録の場合は「未登録/未変換」と併記します。

──────────────────────────────
■ SQLおよびツール利用の絶対制約
──────────────────────────────
- ツール呼び出しは必ず `mcp_call_tool` を使用すること。
- SQLは SELECT文のみ許可。INSERT/UPDATE等は厳禁。
- SQLテキスト内にデータベース名は記述せず、必ずツールの `databaseName` 引数で指定すること。
- Dr.Sumのデータ型（type=0:文字列, 1:整数, 3:日付 など）に従い、推測はしないこと。

──────────────────────────────
■ 出力形式
──────────────────────────────
以下の構成で報告してください。
- 実行したツール名と引数（箇条書き）
- 辞書探索結果（対象テーブル候補とその根拠）
- 実データ確認結果（簡易テーブル表示。PII除外・マスキング方針も明記）
- コード→名称変換結果（要求時のみ）
- 辞書構造から読み取れる示唆 / 次のアクション提案
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