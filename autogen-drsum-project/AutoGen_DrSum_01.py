import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from mcp.client.stdio import stdio_client
from mcp import ClientSession

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core.tools import FunctionTool


def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"環境変数 {name} が未設定です。.env を確認してください。")
    return v


def _optional_env(name: str) -> str | None:
    v = os.getenv(name)
    return v if v else None


def _mcp_result_to_plain_dict(result) -> dict:
    content_out = []
    for c in (result.content or []):
        if hasattr(c, "text"):
            content_out.append(c.text)
        elif hasattr(c, "model_dump"):
            content_out.append(c.model_dump())
        else:
            content_out.append(str(c))
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
    )
    # Dr.Sum 認証情報（ユーザID/パスワード）
    drsum_user = _require_env("DRSUM_USER")
    drsum_password = _require_env("DRSUM_PASSWORD")
    drsum_host = _optional_env("DRSUM_HOST")
    drsum_port = _optional_env("DRSUM_PORT")

    # Dr.Sum MCP Server jar
    mcp_dir = Path(r"c:\drsum-mcp-server")
    jar_path = mcp_dir / "drsum-local-mcp-server-1.1.00.0000.jar"
    if not jar_path.exists():
        raise FileNotFoundError(f"jar が見つかりません: {jar_path}")

    # 注意:
    # jar 側が「起動引数で認証モード/ID/PWを渡す」設計の場合は、ここに追記が必要です。
    # 例: ["java","-jar",..., "--user", drsum_user, "--password", drsum_password]
    cmd = ["java", "-jar", str(jar_path),"--user", drsum_user, "--password", drsum_password]

    async with stdio_client(cmd, cwd=str(mcp_dir)) as (read, write):
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

            async def mcp_call_tool(name: str, arguments_json: str = "{}") -> str:
                try:
                    arguments = json.loads(arguments_json) if arguments_json else {}
                except json.JSONDecodeError as e:
                    return json.dumps(
                        {"isError": True, "content": [f"arguments_json が不正な JSON です: {e}"]},
                        ensure_ascii=False,
                        indent=2,
                    )

                # user/password 等がスキーマ上必要そうなら自動注入
                arguments = _inject_auth_if_needed(name, arguments)

                result = await session.call_tool(name, arguments)
                payload = _mcp_result_to_plain_dict(result)
                return json.dumps(payload, ensure_ascii=False, indent=2)

            tools = [
                FunctionTool(mcp_list_tools, name="mcp_list_tools", description="MCPサーバーが提供するtools一覧を取得する"),
                FunctionTool(
                    mcp_call_tool,
                    name="mcp_call_tool",
                    description="指定したMCP toolを呼び出す。arguments_json は tool の inputSchema に従う JSON 文字列。",
                ),
            ]

            assistant = AssistantAgent(
                name="assistant",
                model_client=model_client,
                tools=tools,
            )

            task = """

あなたは Dr.Sum のローカル MCP サーバー経由でメタデータ/データを取得します（ユーザID/パスワード認証）。
以下の手順で実行し、結果を日本語で整理して出力してください。

1) mcp_list_tools で利用可能な tool 一覧と inputSchema を確認する
2) 「login / connect / open_session」等の認証・接続確立が必要そうな tool があれば、それを最初に実行する
   - user/password 等の引数が必要なら inputSchema に従って指定する（省略時はクライアント側で自動注入される）
3) DB(またはカタログ/スキーマ)一覧を取得する
4) その中の1つを対象に、テーブル一覧とビュー一覧を取得する
5) 代表として最初に見つかったテーブル（またはビュー）から先頭5行を取得する
   - 「SQL実行」系 tool があれば SELECT で先頭5行を取得してよい

注意:
- tool 呼び出しは必ず mcp_call_tool を使うこと
- 取得結果が空/権限エラーの場合は、その旨と次に確認すべき点（認証/権限/対象名/接続先）を出すこと
"""

            await Console(assistant.run_stream(task=task))


if __name__ == "__main__":
    asyncio.run(main())