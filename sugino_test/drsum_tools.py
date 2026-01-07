from mcp import StdioServerParameters, ClientSession
from mcp.client.stdio import stdio_client
import asyncio
import nest_asyncio

# Streamlit等でループが既に回っている場合に対応
nest_asyncio.apply()

# Configuration for the Dr.Sum MCP Server
SERVER_PARAMS = StdioServerParameters(
    command="java",
    args=[
        "-Dfile.encoding=UTF-8",
        "-jar",
        r"C:\drsum-mcp-server\drsum-local-mcp-server-1.0.00.0000.jar",
        "--host=eaasys031s",
        "--port=6001",
        "--user=z21070",
        "--password=*******"
    ],
    env=None
)

async def _run_tool(tool_name: str, arguments: dict = None):
    """MCPツールを非同期で実行する内部関数"""
    try:
        async with stdio_client(SERVER_PARAMS) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments or {})
                if result and result.content:
                    return result.content[0].text
                return "No result from tool."
    except Exception as e:
        return f"Error executing tool {tool_name}: {str(e)}"

def _run_sync(coro):
    """非同期関数を同期的に実行するためのラッパー"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)

# --- 具体的なツール関数 ---

def get_database_list(namePattern: str = None, limit: int = None, offset: int = None):
    """データベース一覧を取得します"""
    args = {}
    if namePattern: args["namePattern"] = namePattern
    if limit: args["limit"] = limit
    if offset: args["offset"] = offset
    return _run_sync(_run_tool("get_database_list", args))

def get_table_list(databaseName: str, namePattern: str = None, tableType: int = None, limit: int = None, offset: int = None):
    """指定したデータベースのテーブル一覧を取得します"""
    args = {"databaseName": databaseName}
    if namePattern: args["namePattern"] = namePattern
    if tableType is not None: args["tableType"] = tableType
    if limit: args["limit"] = limit
    if offset: args["offset"] = offset
    return _run_sync(_run_tool("get_table_list", args))

def get_schema(databaseName: str, tableName: str, limit: int = None, offset: int = None):
    """指定したテーブルのスキーマ情報を取得します"""
    args = {"databaseName": databaseName, "tableName": tableName}
    if limit: args["limit"] = limit
    if offset: args["offset"] = offset
    return _run_sync(_run_tool("get_schema", args))

def execute_select(databaseName: str, selectStatement: str, limit: int = None, offset: int = None):
    """SELECT文を実行して結果を取得します"""
    args = {"databaseName": databaseName, "selectStatement": selectStatement}
    if limit: args["limit"] = limit
    if offset: args["offset"] = offset
    return _run_sync(_run_tool("execute_select", args))

def measure_select(databaseName: str, selectStatement: str, executionCount: int = None, intervalMs: int = None):
    """SELECT文の実行パフォーマンスを計測します"""
    args = {"databaseName": databaseName, "selectStatement": selectStatement}
    if executionCount: args["executionCount"] = executionCount
    if intervalMs: args["intervalMs"] = intervalMs
    return _run_sync(_run_tool("measure_select", args))
