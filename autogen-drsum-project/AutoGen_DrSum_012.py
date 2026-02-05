# 原型はtaskが長いので、Agentに役割を教えて、taskを短くするバージョン 
# Claude版でエラーを何度も修正して動いたものをOpenAIへ移植した
# Azure OpenAIでは、カラムの型 文字列型（Verchar）type=0が、数値型となるので
# `ROLE_INSTRUCTIONS` に 【Dr.Sum スキーマ type コード定義】を追加して正常化した　2026/2/5

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

from requests import session
import truststore

#from asyncio import tools
truststore.inject_into_ssl()

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
# from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_core.tools import FunctionTool

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
    # MCP tool の inputSchema は JSON Schema 互換の想定
    props = schema.get("properties", {}) if isinstance(schema, dict) else {}
    return set(props.keys())

async def main():
    load_dotenv()

    # Azure OpenAI
    deployment = _require_env("DEPLOYMENT_NAME")
    api_key = _require_env("API_KEY")
    endpoint = _require_env("API_ENDPOINT")
    api_version = os.getenv("API_VERSION", "2024-12-01-preview")
    model_name = _require_env("DEFAULT_MODEL_NAME")

    model_client = AzureOpenAIChatCompletionClient(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        azure_deployment=deployment,
        model=model_name,
        temperature=0.2,
    )

    # Dr.Sum 認証情報（ユーザID/パスワード）
    drsum_user = _require_env("DRSUM_USER")
    drsum_password = _require_env("DRSUM_PASSWORD")
    drsum_host = _optional_env("DRSUM_HOST")
    drsum_port = _optional_env("DRSUM_PORT")

    # Configuration for the Dr.Sum MCP Server
    args = [
       "-Dfile.encoding=UTF-8",
       "-jar",
       r"C:\drsum-mcp-server\drsum-local-mcp-server-1.1.00.0000.jar",
       f"--user={drsum_user}", f"--password={drsum_password}",
   ]
    if drsum_host:args.append(f"--host={drsum_host}")
    if drsum_port:args.append(f"--port={drsum_port}")
  
    SERVER_PARAMS = StdioServerParameters(
        command="java",
        args=args,
        env=None,
    )

    # async with stdio_client(cmd, cwd=str(mcp_dir)) as (read, write):
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # tool schema を保持して、呼び出し時に user/password を注入できるようにする
            tool_schema_map: dict[str, dict] = {}

            async def mcp_list_tools() -> str:
                tools = await session.list_tools()
                tool_schema_map.clear()

                tools_out = []
                for t in tools.tools:
                    schema = t.inputSchema or {}
                    tool_schema_map[t.name] = schema

                    props = list((schema.get("properties") or {}).keys()) if isinstance(schema, dict) else []
                    required = schema.get("required", []) if isinstance(schema, dict) else []
                    tools_out.append({
                        "name": t.name,
                        "description": t.description,
                        "properties": props,
                        "required": required,
                    })

                payload = {"tools": tools_out}
                return json.dumps(payload, ensure_ascii=False, indent=2)

            def _inject_auth_if_needed(tool_name: str, args: dict) -> dict:
                schema = tool_schema_map.get(tool_name) or {}
                props = _schema_properties(schema)

                # よくあるキー名候補に対して自動注入（無い場合のみ）
                candidates_user = ["user", "userid", "userId", "username", "loginId", "login_id"]
                candidates_pass = ["password", "pass", "pwd"]

                if any(k in props for k in candidates_user):
                    for k in candidates_user:
                        if k in props and k not in args:
                            args[k] = drsum_user
                            break

                if any(k in props for k in candidates_pass):
                    for k in candidates_pass:
                        if k in props and k not in args:
                            args[k] = drsum_password
                            break

                # 接続先が必要な tool の場合のみ注入（設定してあるときだけ）
                if drsum_host:
                    for k in ["host", "server", "hostname", "endpoint"]:
                        if k in props and k not in args:
                            args[k] = drsum_host
                            break
                if drsum_port:
                    for k in ["port"]:
                        if k in props and k not in args:
                            # 数値要求の可能性があるため int 変換を試みる
                            try:
                                args[k] = int(drsum_port)
                            except ValueError:
                                args[k] = drsum_port
                            break

                return args

            async def mcp_call_tool(
                name: str,
                arguments: dict | None = None,
                arguments_json: str | None = None
            ) -> str:
                # 1) arguments を確実に dict にする（None を残さない）
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

                # 2) execute_select の引数名ゆらぎを吸収（sql -> selectStatement）
                if name == "execute_select" and isinstance(arguments, dict):
                    if "selectStatement" not in arguments and "sql" in arguments:
                        arguments["selectStatement"] = arguments.pop("sql")

                # 3) 認証/host/port の自動注入
                arguments = _inject_auth_if_needed(name, arguments)

                # 4) MCP tool 呼び出し
                result = await session.call_tool(name, arguments)

                # 5) MCP戻りを整形して返す（json文字列はdict化済みの前提）
                payload = _mcp_result_to_plain_dict(result)
                return json.dumps(payload, ensure_ascii=False, indent=2)

            agent_tools = [
                FunctionTool(mcp_list_tools, name="mcp_list_tools", description="MCPサーバーが提供するtools一覧を取得する"),
                FunctionTool(
                    mcp_call_tool,
                    name="mcp_call_tool",
                    description="指定したMCP toolを呼び出す。arguments(dict)推奨。互換として arguments_json(文字列JSON) も可。",
                ),
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

            client = model_client
            assistant = AssistantAgent(
                name="assistant",
                model_client=client,  # AzureOpenAIChatCompletionClient
                tools=agent_tools,           # mcp_list_tools / mcp_call_tool
                system_message=ROLE_INSTRUCTIONS,  # ← 役割を固定
                reflect_on_tool_use=True,   # ★ツール結果を見て次の手に進む
                max_tool_iterations=8,      # ★ツール呼び出し反復回数（DB→table→schema→select なら4以上）
            )

            task = """
目的：DB一覧を取得し、先頭のDBのテーブル一覧(tableType=0)を取得し、先頭テーブルのスキーマと先頭5行を表示してください。

手順：
1) mcp_call_tool で get_database_list（limit=50）
2) databases[0] を databaseName として get_table_list（tableType=0, limit=2000）
3) tables[0].name を tableName として get_schema（limit=999）_
4) mcp_call_tool で execute_select を呼ぶ。arguments にはdatabaseName と selectStatement（SQL文）と limit を渡す。
出力：テーブル一覧、先頭テーブルの列定義、先頭5行を日本語で整理して報告
"""

            await Console(assistant.run_stream(task=task))


if __name__ == "__main__":
    asyncio.run(main())