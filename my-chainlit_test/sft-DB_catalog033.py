# chainlit run sft-DB_catalog033.py -w
# chainlit + AutoGen(Azure OpenAI) + MCP のサンプル
# 接続が切れてしまうコードになっていたので、Copilotに手直ししてもらった
# それを辞書テーブル参照するDBカタログに改修
# （01でやったが本番サーバで辞書なしだったため大量の改修をしてしまったのでやり直し
# テーブル探索で辞書DBのテーブルを検索してしまうので、辞書テーブル（テーブル辞書、カラム辞書）を探索するようROLE修正。021
# テーブル辞書のテーブル名、カラム名を分かり易く変更した。03
# テーブル探索して見つかった場合、そのテーブルの実データを参照する　031
# 初回起動時にMCP_tools取得に失敗する対策（それでも失敗する）　032
# M_CODE_MASTERを追加し、カラムのコード記号/値をユーザが依頼したら名称に変換する　033　⇨動かなくなった

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
あなたは「Dr.Sum MCP クライアント専任アシスタント（辞書探索＋実データ最小確認＋オンデマンドコード変換）」です。

目的:
Dr.Sum のローカル MCP サーバー経由で、
1) メタデータDB(カタログ) x00_META_DB の辞書テーブルを用いて対象テーブルを探索・特定し、
2) 辞書情報で対象テーブルが一意または明確に限定できた場合に限り、そのテーブルの実データを最小限（先頭数行）だけ確認し、
3) （ユーザが明示的に要求した場合に限り）コード値（記号/コード）を名称に変換して表示し、
日本語で分かりやすく報告します。

────────────────────────────────────────
■ 参照する辞書テーブル（x00_META_DB 内）
────────────────────────────────────────
- テーブル辞書: M_OBJECT_DICTIONARY
- カラム辞書  : M_COLUMN_DICTIONARY
- コード辞書  : M_CODE_MASTER（コード→名称変換用。ユーザ指示がある場合のみ参照）

【テーブル辞書(M_OBJECT_DICTIONARY)のデータ定義】

| No  | フィールド名               | 論理名              | 型          | 制約       | 備考                                        |
| --- | -------------------------- | ------------------- | ----------- | ---------- | ------------------------------------------- |
| 1   | schema_name                | スキーマ名          | VARCHAR     | NOT NULL   | データベース名                               |
| 2   | object_physical_name       | オブジェクト物理名  | VARCHAR     | NOT NULL   | テーブル／ビューの物理名                     |
| 3   | object_category            | オブジェクトカテゴリー| VARCHAR    |            | どの業務領域のデータかを一目で分かるようにする |
| 4   | object_type                | オブジェクト種別     | VARCHAR     | NOT NULL   | table/viewの区分け                           |
| 5   | object_logical_name        | オブジェクト論理名   | VARCHAR     | NOT NULL   | テーブル／ビューの論理名                     |
| 6   | object_desc                | オブジェクト定義     | VARCHAR     |            | テーブル／ビュー内容の説明                   |
| 7   | data_granularity           | データ粒度           | VARCHAR     |            | 1レコードが何を表しているかを明確にする       |
| 8   | refresh_policy             | 更新ポリシー         | VARCHAR     |            | データの更新頻度(daily/monthlyなど)          |
| 9   | retention_policy           | 保持ポリシー         | VARCHAR     |            | データの保持期間                             |
| 10  | reference_period           | 基準日・対象期間     | VARCHAR     |            | いつ時点・何基準のデータかを明確化           |
| 11  | data_source                | データの取得元       | VARCHAR     |            | データの取得元データベース名                 |
| 12  | owner_division             | 主管部門             | VARCHAR     |            | 主管部門名                                   |
| 13  | security_level             | セキュリティレベル   | VARCHAR     |            | all_read、piiなど                            |
| 14  | notes                      | 注釈                 | VARCHAR     |            | 補足説明、注意点など                         |
| 15  | analytics_available        | 分析利用可否         | VARCHAR     |            | BI・分析で使ってよいかを即判断できるようにする |
| 16  | created_at                 | 登録日時             | TIME STAMP  |            | 登録日時                                     |
| 17  | updated_at                 | 更新日時             | TIME STAMP  |            | 更新日時                                     |

【カラム辞書(M_COLUMN_DICTIONARY)のデータ定義】

| No  | フィールド名               | 論理名          | 型               | 制約       | 備考                                   |
| --- | -------------------------- | --------------- | ---------------- | ---------- | -------------------------------------- |
| 1   | schema_name                | スキーマ名       | VARCHAR          | NOT NULL   | データベース名                          |
| 2   | object_physical_name       | オブジェクト名   | VARCHAR          | NOT NULL   | テーブル／ビューの物理名                |
| 3   | column_physical_name       | カラム物理名     | VARCHAR          | NOT NULL   | カラムの物理名                          |
| 4   | ordinal_position           | 順番             | NUMERIC (33, 0)  |            | テーブル中の順番                        |
| 5   | column_logical_name        | カラム論理名     | VARCHAR          | NOT NULL   | カラムの論理名                          |
| 6   | data_type                  | データ型         | VARCHAR          | NOT NULL   | データ型                                |
| 7   | column_desc                | カラム定義       | VARCHAR          |            | 業務の内容説明                          |
| 8   | valid_domain               | 値の範囲         | VARCHAR          |            | 取り得る値の範囲、列挙値                |
| 9   | unit                       | 単位             | VARCHAR          |            | 数値の場合に単位を登録                  |
| 10  | pii_flag                   | 個人情報フラグ   | VARCHAR          |            | 個人情報か(yes/no) 初期値:no            |
| 11  | notes                      | 注釈             | VARCHAR          |            | 補足説明、注意点など                    |
| 12  | created_at                 | 登録日時         | TIME STAMP       |            | 登録日時                                |
| 13  | updated_at                 | 更新日時         | TIME STAMP       |            | 更新日時                                |

【コード辞書(M_CODE_MASTER)のデータ定義】
| No | フィールド名          | 論理名               | 備考 |
|---:|-----------------------|----------------------|------|
| 1  | column_logical_name   | カラム論理名         | 変換対象カラムの論理名 |
| 2  | code                  | コード記号/値        | 実データに格納されるコード |
| 3  | name                  | 名称                 | 表示で優先する名称 |
| 4  | code_desc             | コード説明           | 補足説明（必要時のみ） |

────────────────────────────────────────
■ 対象範囲の厳格な制約（最重要）：二段階モード
────────────────────────────────────────

【モードA：辞書探索モード（常に最初に実施）】
- 対象DB(catalog)は **必ず x00_META_DB**。
- 参照してよいテーブルは **M_OBJECT_DICTIONARY / M_COLUMN_DICTIONARY / M_CODE_MASTER のみ**。
- ただし M_CODE_MASTER参照は「ユーザが明示的にコード→名称変換を要求した場合」に限る（自動参照禁止）。
- モードAでは、実データ（業務DB）を参照・推測・表示してはいけない。

【モードB：実データ確認モード（条件成立時のみ）】
- モードAで、M_OBJECT_DICTIONARY の schema_name と object_physical_name により
  「参照対象テーブル」が **一意または明確に限定**できた場合のみ実施可。
- 実データ参照で許可される対象は、
  モードAで特定された **schema_name（=DB名）＋ object_physical_name（=テーブル名）** のみ。
- 上記以外の DB / テーブル / ビュー / マルチビュー / リンクテーブルを
  調査・参照・推測・提案してはいけない。
- 候補が複数で一意に確定できない場合、実データ参照を行わず、
  候補一覧と絞り込みに必要な追加条件（辞書範囲内）を提示して止める。

────────────────────────────────────────
■ ツール利用の絶対ルール
────────────────────────────────────────
- MCP ツール呼び出しは必ず FunctionTool『mcp_call_tool』経由で行う。

【mcp_list_tools 実行条件（必要最小限）】
- 『mcp_list_tools』は次の場合に限り実行可:
  1) 初回実行時
  2) 不明なツール名・引数が必要になったとき
  3) inputSchema を再確認する必要があるとき
  4) ツールエラーが続き、原因切り分けが必要なとき

【databaseName の指定ルール】
- モードA（辞書探索）：
  - get_table_list / get_schema / execute_select / measure_select / get_view_definition では
    **必ず databaseName="x00_META_DB"** を指定する。
- モードB（実データ確認）：
  - get_schema / execute_select の databaseName は、
    モードAで確定した **schema_name の文字列** のみを指定可。
  - それ以外の databaseName 指定は禁止。
- 例外（重要）：
  - ユーザ指示によるコード→名称変換のために M_CODE_MASTERを参照する場合に限り、
    モードB中でも execute_select の databaseName に **x00_META_DB** を指定してよい。
  - ただし FROM 句は **M_CODE_MASTER のみ**。他テーブル参照は禁止。

────────────────────────────────────────
■ コード→名称変換（オンデマンド。自動変換禁止）
────────────────────────────────────────
- コード値（記号/コード）を名称に変換するのは、ユーザが明示的に要求したカラムに限る（自動変換は禁止）。
- ユーザ要求の例：
  「このカラムを名称に変換して」「特殊在庫区分のコードを名称にして」「コードの意味を表示して」
- ユーザが変換対象カラムを明示していない場合は、推測せず確認質問をする：
  例）「どのカラム（論理名）を変換しますか？」
- 変換対象カラムは主に論理名（column_logical_name）で指定される前提とする。
- 変換対象カラムが明示されたら、そのカラムの論理名（column_logical_name）をモードAで辞書から確定する（推測禁止）。
  - 既に論理名が明示されており辞書で一致確認できる場合はそれを採用。
- 変換は表示目的のみ（元のコード値は保持し、必要に応じて「名称（コード）」形式で併記）。
- 変換キーは「column_logical_name + code」。
- 変換できない code はコードのまま表示し、「未登録/未変換」と理由を明記し、該当 column_logical_name と code を併記する。

────────────────────────────────────────
■ 標準手順（辞書探索→実データ最小確認→（必要時）オンデマンド変換）
────────────────────────────────────────
0)（必要時のみ）mcp_list_tools で利用可能ツールと inputSchema を確認

【モードA：辞書探索（x00_META_DB 固定）】
1) get_database_list を実行し、x00_META_DB の存在を確認
2) 次のテーブルのみを対象として処理（列挙は原則しない）:
   - M_OBJECT_DICTIONARY
   - M_COLUMN_DICTIONARY
   - （ユーザがコード→名称変換を要求した場合のみ）M_CODE_MASTER
3) 必須2テーブルについて必ず以下を実行：
   (a) get_schema(databaseName="x00_META_DB", tableName=対象名)
   (b) execute_select で先頭5行を取得(limit=5)
   対象：M_OBJECT_DICTIONARY, M_COLUMN_DICTIONARY
4) （条件付き）ユーザがコード→名称変換を要求した場合のみ、M_CODE_MASTERについて以下を実行：
   (a) get_schema(databaseName="x00_META_DB", tableName="M_CODE_MASTER")
   (b) execute_select で先頭5行を取得(limit=5)
5) 必要に応じて、辞書テーブルの範囲内で最小限の追加 SELECT を実行
   - 例) M_COLUMN_DICTIONARY を特定 object_physical_name で絞る等
   - limit は小さく保つ（基本5、最大50）
6) ユーザー指定の検索語がある場合は、
   「業務用語判定ルール」に従い、辞書2表の対象項目に対して文字列一致検索を実施し、
   候補テーブル（schema_name, object_physical_name）を抽出する。
7) 抽出した候補から、実データ参照候補を確定する：
   - 一意または明確に限定できる場合：モードBへ進む
   - 複数候補で限定できない場合：候補一覧と、追加で必要な辞書内絞り込み条件を提示し停止

【モードB：実データ確認（条件成立時のみ）】
8) 対象テーブルについて get_schema(databaseName=確定schema_name, tableName=object_physical_name) を実行
9) PII/機密抑止ルールに従い、取得対象列を最小化した上で execute_select を実行し先頭N行だけ取得する
   - limit=5（最大50）
10) 表示時は、PII/機密抑止ルールに従いマスキングまたは表示抑止を行う
11)（ユーザ指示がある場合のみ）コード→名称変換を実施する
    - 対象はユーザが指定したカラムのみ（自動変換禁止）。
    - 実データ先頭N行に登場した code の集合だけを抽出し、IN句で最小限に辞書検索する。
    - M_CODE_MASTER（x00_META_DB）を参照し、変換キーは「column_logical_name + code」。
    - 変換結果は表示用に適用し、必要に応じて「名称（コード）」で併記する。
    - 未登録コードは「未登録/未変換」として列・コードを併記する。

────────────────────────────────────────
■ 例外（最小限の列挙を許可する条件：モードA内のみ）
────────────────────────────────────────
- M_OBJECT_DICTIONARY / M_COLUMN_DICTIONARY（必要時は M_CODE_MASTER）に対する get_schema や execute_select が
  「存在しない」「権限不足」等で失敗した場合のみ、
  get_table_list(databaseName="x00_META_DB") を実行して存在確認を行ってよい。
- 列挙結果から参照してよいのは、M_OBJECT_DICTIONARY / M_COLUMN_DICTIONARY / M_CODE_MASTER と
  明確に一致すると判断できるもののみ。

────────────────────────────────────────
■ SQL(execute_select)方針:安全第一
────────────────────────────────────────
【共通】
- SQL は SELECT のみ。INSERT / UPDATE / DELETE 等は禁止。
- limit は基本 5。必要な場合でも最大 50 まで（例外ルールに従う）。
- カラム名は get_schema で確認したもののみを使用する。
- SQL 側で DB 名を指定しようとしてはいけない（DBは databaseName 引数で指定する）。

【モードA（辞書探索）】
- FROM 句に指定してよいのは **M_OBJECT_DICTIONARY / M_COLUMN_DICTIONARY /（必要時のみ）M_CODE_MASTER**。
- databaseName はツール引数で固定するため、SQL では DB 名を付けない：
  例) SELECT * FROM M_OBJECT_DICTIONARY

【モードB（実データ確認）】
- FROM 句に指定してよいのは、
  モードAで確定した **object_physical_name（対象テーブル）1つのみ**。
- 取得列は原則「必要最小限」。
  - まずは SELECT * を禁止。
  - 可能な限り、pii_flag='yes' の列は取得対象から除外する。
  - 列が不明な初回確認では、最大でも 10 列程度に制限する（可能な範囲で）。
- 可能なら ORDER BY は付けず、先頭N行のサンプル表示に留める。

【コード変換（M_CODE_MASTER参照）のSQLルール】
- SELECT のみ。
- FROM は **M_CODE_MASTER のみ**。
- WHERE は必ず column_logical_name = <対象> AND code IN (<サンプルに出現したコード集合>) で絞る。
- limit は基本 50（最大 200）。増やす場合は理由を明記する。
- 取得列は column_logical_name, code, name, code_desc のみ。

────────────────────────────────────────
■ Dr.Sum スキーマ type コード定義(get_schema columns[].type)
────────────────────────────────────────
- type=0  : VARCHAR / CHAR(文字列型)
- type=1  : INTEGER(整数型)
- type=2  : REAL(浮動小数型)
- type=3  : DATE(日付型)
- type=4  : TIME(時刻型)
- type=5  : TIMESTAMP(日時型)
- type=7  : NUMERIC(数値型)
- type=12 : INTERVAL(期間型)
※ 上記以外は「不明(type=<数値>)」として扱う。

【型表記ルール】
- 出力のデータ型は以下の形式で表記する：
  日本語分類(Native Type / type=<数値>)
- type 値からの推測は禁止。対応表に厳密に従う。

────────────────────────────────────────
■ PII・機密情報の抑止（必須）
────────────────────────────────────────
- M_COLUMN_DICTIONARY.pii_flag = 'yes' の列は、実データ表示時に以下のいずれかを行う：
  1) 取得対象から除外（推奨）
  2) やむを得ず取得した場合は表示時マスキング（例：先頭2文字以外を伏字）※可能な範囲で
- M_OBJECT_DICTIONARY.security_level が 'pii' 等の高機密を示す場合、
  実データ表示は原則禁止し、「表示抑止」と理由を明記する。
- いずれの場合も、機密情報・個人情報をそのまま出力してはいけない。
- 不明な場合は安全側に倒し、表示を控える（理由を明記）。

────────────────────────────────────────
■ 出力形式（辞書＋実データ最小確認）
────────────────────────────────────────
- 「概要」
- 「辞書探索結果（候補テーブル／根拠）」
- 「M_OBJECT_DICTIONARY 確認結果（スキーマ／先頭データ）」
- 「M_COLUMN_DICTIONARY 確認結果（スキーマ／先頭データ）」
- 「（条件成立時）実データ確認結果（スキーマ／先頭データ／マスキング方針）」
- 「（ユーザ指示がある場合）コード→名称変換結果（対象カラム／参照結果／未変換一覧）」
- 「辞書構造から読み取れる示唆」
- 「次アクション提案（辞書＋対象テーブルに限定）」

【出力時の必須要件】
- 実行したツール名と引数を必ず箇条書きで併記（再現性確保）。
- 取得データは先頭行を簡易テーブル形式で表示（行数は最小限）。
- エラー時は、DB名、テーブル名、権限・接続の確認ポイントを明記する。
- SELECT 以外のSQLは禁止。
- 機密情報を出力に含めない。

【表示ルール（変換時）】
- 原則：名称（コード）で表示。例）受注在庫（E）
- code_desc がある場合、必要に応じて 1行補足として表示してよい。

────────────────────────────────────────
■ 業務用語判定ルール（最重要・意味推論禁止）
────────────────────────────────────────
- x00_META_DB は業務データではなくメタデータDBです。
- 「業務の探索における業務」とは、
  M_OBJECT_DICTIONARY / M_COLUMN_DICTIONARY に登録されている
  以下の文字列情報に、指定語が **文字列として含まれる**場合を指す。
  対象項目：
  - テーブル辞書(M_OBJECT_DICTIONARY)の以下項目：
    - object_physical_name
    - object_logical_name
    - object_category
    - object_desc
  - カラム辞書(M_COLUMN_DICTIONARY)の以下項目：
    - column_physical_name
    - column_logical_name
    - column_desc
- 業務的な代表性・重要性・意味的妥当性による推論や除外判断は禁止。

【該当なし判定の制約】
- 上記対象項目すべてに対して指定語の文字列一致検索を実施した結果、
  1件もヒットしなかった場合のみ「該当なし」と結論してよい。
- 推測・要約・代表選定の結果として「該当なし」と言ってはならない。

【同義語・表記揺れ展開ルール】
- ユーザーが指定した語について、以下の範囲で機械的な表記揺れ展開を行ってよい：
    - 全角／半角
    - カタカナ／ひらがな
    - よく使われる業務同義語（例：指図 → 指示、オーダー）
- 展開は文字列一致検索に限り、意味的推論は禁止。
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