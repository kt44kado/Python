# chainlit run sft-DB_catalog036.py
# chainlit + AutoGen(Azure OpenAI) + MCP のサンプル
# 接続が切れてしまうコードになっていたので、Copilotに手直ししてもらった
# それを辞書テーブル参照するDBカタログに改修
# （01でやったが本番サーバで辞書なしだったため大量の改修をしてしまったのでやり直し
# テーブル探索で辞書DBのテーブルを検索してしまうので、辞書テーブル（テーブル辞書、カラム辞書）を探索するようROLE修正。021
# テーブル辞書のテーブル名、カラム名を分かり易く変更した。03
# テーブル探索して見つかった場合、そのテーブルの実データを参照する　031
# 初回起動時にMCP_tools取得に失敗する対策（それでも失敗する）　032
# M_CODE_MASTERを追加し、カラムのコード記号/値をユーザが依頼したら名称に変換する　033　⇨動かなくなった
# ROLEをシンプルに変換した　034 ただし、実データを表示しない
# ROLEのstep2を修正したら動いた。またコードを名称に変更表示もした。
# objectのクエリーを表示できるようにした　036
# MCPをURL接続に変更　04

import asyncio
import json
import os
import sys
import requests
import chainlit as cl
from dotenv import load_dotenv
import truststore

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core.tools import FunctionTool

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

class MCPBridgeContext:
    """Bridge Server経由でMCPツールを呼び出すコンテキスト"""
    def __init__(self, bridge_url, drsum_user, drsum_password, drsum_host, drsum_port):
        self.bridge_url = bridge_url.rstrip("/")
        self.drsum_user = drsum_user
        self.drsum_password = drsum_password
        self.drsum_host = drsum_host
        self.drsum_port = drsum_port
        self.lock = asyncio.Lock()

    def _build_auth(self) -> dict:
        auth = {"user": self.drsum_user, "password": self.drsum_password}
        if self.drsum_host:
            auth["host"] = self.drsum_host
        if self.drsum_port:
            try:
                auth["port"] = int(self.drsum_port)
            except ValueError:
                pass
        return auth

    async def list_tools_json(self) -> str:
        """Bridge Serverから利用可能なツール一覧を取得する"""
        # Bridge Serverが /mcp/execute で list_tools をサポートしている前提の簡易実装
        # 実際には固定のリストを返しても良いですが、ここではBridge経由で取得を試みます
        body = {
            **self._build_auth(),
            "method": "tools/list",
            "params": {}
        }
        try:
            loop = asyncio.get_event_loop()
            res = await loop.run_in_executor(None, lambda: requests.post(f"{self.bridge_url}/mcp/execute", json=body, timeout=30))
            return json.dumps(res.json(), ensure_ascii=False, indent=2)
        except Exception as e:
            return json.dumps({"error": f"ツール一覧の取得に失敗しました: {e}"}, ensure_ascii=False)

    async def call_tool_json(self, name: str, arguments: dict | None = None) -> str:
        """Bridge Serverの /mcp/execute を呼び出す"""
        if arguments is None:
            arguments = {}

        # 既存の引数名補正 (sql -> selectStatement)
        if name == "execute_select" and "sql" in arguments:
            arguments["selectStatement"] = arguments.pop("sql")

        body = {
            **self._build_auth(),
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments,
            },
        }

        async with self.lock:
            try:
                loop = asyncio.get_event_loop()
                # ネットワークI/Oを非同期で実行
                res = await loop.run_in_executor(
                    None, 
                    lambda: requests.post(f"{self.bridge_url}/mcp/execute", json=body, timeout=120)
                )
                res.raise_for_status()
                return json.dumps(res.json(), ensure_ascii=False, indent=2)
            except Exception as e:
                return json.dumps({"isError": True, "content": [str(e)]}, ensure_ascii=False, indent=2)

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

    # 環境変数からBridge ServerとDr.Sum情報を取得
    bridge_url = os.getenv("BRIDGE_URL", "http://eaasys022s:3000")
    drsum_user = _require_env("DRSUM_USER")
    drsum_password = _require_env("DRSUM_PASSWORD")
    drsum_host = _optional_env("DRSUM_HOST")
    drsum_port = _optional_env("DRSUM_PORT")

    # Bridge接続用コンテキストの作成
    mcp_ctx = MCPBridgeContext(bridge_url, drsum_user, drsum_password, drsum_host, drsum_port)

    agent_tools = [
        FunctionTool(mcp_ctx.list_tools_json, name="mcp_list_tools", description="MCPサーバーが提供するtools一覧を取得する"),
        FunctionTool(mcp_ctx.call_tool_json, name="mcp_call_tool", description="指定したMCP toolを呼び出す。arguments(dict)形式で指定すること。"),
    ]

    # プロンプト（ROLE_INSTRUCTIONS）は元の定義を維持
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
4. 情報系DBクエリ (クエリ辞書): object_physical_name, query (オブジェクト作成時のSQL)

──────────────────────────────
■ 動作ステップと厳格なルール
──────────────────────────────
【ステップ1：辞書探索（意味推論は禁止）】
1. ユーザーの指定語をもとに、辞書に対して `execute_select` で文字列検索を行い、対象テーブルを探索します。
2. 同義語・表記揺れ（全角半角、カナひら等）の展開は許可しますが、業務的意味の推論はせず、文字列一致を正とします。
3. 候補が複数ある場合は一覧と絞り込み条件を提示して終了します。
4. 対象が1つに明確に特定できた場合、`security_level` を確認し、'pii'等の高機密であれば「実データ表示抑止」として理由を報告し、ここで終了します。

【ステップ1.5：定義クエリの表示（ユーザ要求時のみ）】
1. ユーザから「クエリを見せて」「定義を確認したい」等の要求があった場合、特定した `object_physical_name` をキーに `情報系DBクエリ` テーブルを検索し、`query` カラムの内容をそのまま提示してください。

【ステップ2：実データ最小確認（条件成立時のみ）】
1. ステップ1で特定した 接続先データベース名（例：情報系DB） を databaseName に指定する。
2. 事前に get_schema（可能なら）で object_physical_name の存在を確認し、必要なら schema 修飾（例：dbo.VWMMT2100）を確定する。
3. execute_select は PII除外カラムを明示列挙し、先頭5件（最大50件）のみ取得する。

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
- オブジェクト定義クエリ（要求時のみ。Markdownのコードブロックで表示）
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
        max_tool_iterations=10,
    )

    cl.user_session.set("agent", agent)
    #cl.user_session.set("mcp_ctx", mcp_ctx)
    try:
        res = requests.get(f"{bridge_url}/health", timeout=5)
        if res.status_code == 200:
            await cl.Message(content=f"✅ Bridge Server ({bridge_url}) に接続されました。").send()
    except:
        await cl.Message(content=f"⚠️ Bridge Server ({bridge_url}) への接続に失敗しました。URLを確認してください。").send()

@cl.on_message
async def run_conversation(message: cl.Message):
    agent = cl.user_session.get("agent")
    result = await agent.run(task=message.content)
    await cl.Message(content=result.messages[-1].content, author="Assistant").send()