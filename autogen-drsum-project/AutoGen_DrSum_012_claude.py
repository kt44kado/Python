# 原型はtaskが長いので、Agentに役割を教えて、taskを短くするバージョン <--エラーが出て動かない　2026/2/3
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

import truststore
truststore.inject_into_ssl()

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
# from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_core.tools import FunctionTool


def require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"環境変数 {name} が未設定です。.env を確認してください。")
    return v


def optional_env(name: str) -> str | None:
    v = os.getenv(name)
    return v if v else None

def try_parse_json(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x

def mcp_result_to_plain_dict(result) -> dict:
    content_out = []
    for c in (result.content or []):
        if hasattr(c, "text"):
            # MCPが返す text がJSON文字列ならdict/listにパースして返す
            content_out.append(try_parse_json(c.text))
        elif hasattr(c, "model_dump"):
            content_out.append(c.model_dump())
        else:
            content_out.append(try_parse_json(str(c)))
    return {"isError": bool(getattr(result, "isError", False)), "content": content_out}

def schema_properties(schema: dict) -> set[str]:
    # MCP tool の inputSchema は JSON Schema 互換の想定
    props = schema.get("properties", {}) if isinstance(schema, dict) else {}
    return set(props.keys())

async def main():
    load_dotenv()

    # Claude
    def get_model_client():
    # 環境変数 ANTHROPIC_API_KEY が設定されている前提
        return AnthropicChatCompletionClient(
        model="claude-sonnet-4-5-20250929",
        temperature=0.2,
    )

    # Dr.Sum 認証情報（ユーザID/パスワード）
    drsum_user = require_env("DRSUM_USER")
    drsum_password = require_env("DRSUM_PASSWORD")
    drsum_host = optional_env("DRSUM_HOST")
    drsum_port = optional_env("DRSUM_PORT")

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
                for t in tools.tools:
                    tool_schema_map[t.name] = t.inputSchema or {}
                payload = {
                    "tools": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "inputSchema": t.inputSchema,
                        }
                        for t in tools.tools
                    ]
                }
                return json.dumps(payload, ensure_ascii=False, indent=2)

            def inject_auth_if_needed(tool_name: str, args: dict) -> dict:
                schema = tool_schema_map.get(tool_name) or {}
                props = schema_properties(schema)

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
                arguments_json: str | None = None) -> dict:
                # 互換：arguments が無ければ arguments_json を読む
                if arguments is None:
                    if arguments_json:
                        try:
                            arguments = json.loads(arguments_json)
                        except json.JSONDecodeError as e:
                            return {"isError": True, "content": [f"arguments_json が不正な JSON です: {e}"]}
                    else:
                        arguments = {} 
  
                # user/password 等がスキーマ上必要そうなら自動注入
                arguments = inject_auth_if_needed(name, arguments)
                result = await session.call_tool(name, arguments)
                return mcp_result_to_plain_dict(result)

            tools = [
                FunctionTool(mcp_list_tools, name="mcp_list_tools", description="MCPサーバーが提供するtools一覧を取得する"),
                FunctionTool(
                    mcp_call_tool,
                    name="mcp_call_tool",
                    description="指定したMCP toolを呼び出す。arguments は inputSchema に従う dict。互換として arguments_json(文字列JSON) も可。",
                ),
            ]

            ROLE_INSTRUCTIONS = """
あなたは「Dr.Sum MCP クライアント専任アシスタント」です。
目的：Dr.Sum のローカル MCP サーバー経由で、メタデータの把握と少量データの確認を行い、結果を日本語で分かりやすく報告します。

【ツール利用ルール】
- MCPのツール呼び出しは必ず FunctionTool『mcp_call_tool』経由で行うこと。
- 最初に『mcp_list_tools』で利用可能なツール名と inputSchema を把握すること。
- 認証／接続確立が必要そうなツールが存在する場合は、最初にそれを実行すること。
  - user/password/host/port はクライアントから自動注入される（inputSchema にキーがある場合）。
- 標準手順：
  1) DB（またはカタログ／スキーマ）一覧を取得
  2) 1つ選んでテーブル一覧（tableType=0）とビュー一覧（tableType=1）を取得
  3) 代表として先頭のテーブル（またはビュー）から先頭5行を確認（execute_select で limit=5）
  4) 必要に応じて get_schema / get_view_definition で補足情報を取得

【出力形式】
- 「概要」→「詳細結果」→「次アクション提案」の順で日本語で整理。
- 実行したツール名と引数を箇条書きで併記（再現性のため）。
- 取得データの先頭行はテーブル風に整形（可能な範囲で）。
- エラーや空結果時は、認証／権限／対象名／接続先などの確認ポイントを明記。

【注意】
- SELECT 以外のSQLは実行しない（execute_select は読み取り専用）。
- トークン消費を抑えるため、limit は小さめ（まず 5）。
- 機密情報（パスワード等）を出力に含めない。

- テーブル一覧取得後は、必ず「先頭テーブル」について
    (a) get_schema で列定義を取得し、(b) execute_select で先頭5行（limit=5）を取得すること。
- テーブル名に日本語や記号が含まれる可能性があるため、SELECTでは原則としてダブルクォートで囲む（例: SELECT * FROM "テーブル名"）。
    もしエラーになる場合のみ、クォート無しも試す。
"""
            client = get_model_client()
            assistant = AssistantAgent(
                name="assistant",
                model_client=client,  # AzureOpenAIChatCompletionClient
                tools=tools,           # mcp_list_tools / mcp_call_tool
                system_message=ROLE_INSTRUCTIONS,  # ← 役割を固定
                # instructions=ROLE_INSTRUCTIONS,  # ← 役割（system）を固定
            )

            task = """
目的：DB一覧を取得し、先頭のDBのテーブル一覧(tableType=0)を取得し、先頭テーブルのスキーマと先頭5行を表示してください。

手順：
1) mcp_call_tool で get_database_list（limit=50）
2) databases[0] を databaseName として get_table_list（tableType=0, limit=2000）
3) tables[0].name を tableName として get_schema（limit=999）_
4) execute_select で SELECT * FROM "tableName" を実行し、limit=5
    もし SQL エラーなら SELECT * FROM tableName（クォート無し）も試す
出力：テーブル一覧、先頭テーブルの列定義、先頭5行を日本語で整理して報告
"""

            await Console(assistant.run_stream(task=task))


if name == "main":
#if __name__ == "__main__":
    asyncio.run(main())