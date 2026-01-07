#======================================
# ライブラリ
#======================================
import os
import json
from autogen import AssistantAgent, UserProxyAgent, GroupChat
from autogen.agentchat.groupchat import GroupChatManager


from dotenv import load_dotenv
import drsum_tools

#======================================
# 環境変数の読み込み
#======================================
load_dotenv()

#======================================
# Azure OpenAI LLMの設定
#======================================
llm_config = {
    "config_list": [
        {
            "model": os.getenv("DEPLOYMENT_NAME"),
            "api_type": "azure",
            "api_key": os.getenv("API_KEY"),
            "base_url": os.getenv("API_ENDPOINT"),
            "api_version": os.getenv("API_VERSION"),
        }
    ]
}

#======================================
# 終了条件の判定関数
#======================================
def is_termination_msg(msg):
    """
    会話を終了すべきかを判定する関数
    オーケストレーターが最終的な提案をまとめた場合に終了
    """
    if not isinstance(msg, dict):
        return False
    
    content = msg.get("content", "")
    if not isinstance(content, str):
        return False
    
    # 発言者がオーケストレーターの場合のみチェック
    sender = msg.get("name", "")
    if sender != "オーケストレーター":
        return False
    
    # 終了を示すキーワードをチェック
    termination_keywords = [
        "以上です",
        "以上となります",
        "ご確認ください",
        "お知らせください",
        "お申し付けください",
        "教えてください",
        "回答は以上",
        "調査完了"
    ]
    
    # メッセージの最後の部分に終了キーワードが含まれているかチェック
    content_lower = content.strip()
    for keyword in termination_keywords:
        if keyword in content_lower[-50:]:  # 最後の50文字をチェック
            return True
    
    return False

#======================================
# UserProxyAgent
#======================================
user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    code_execution_config={"use_docker": False},
    max_consecutive_auto_reply=30,
    is_termination_msg=is_termination_msg,
    )

#======================================
# Agentの取得
#======================================
def get_agent_names(selected_agents):
    """User, オーケストレーターを除いたエージェント名リストを返す"""
    return [name for name in selected_agents if name not in ("オーケストレーター", "User")]

#======================================
# オーケストレーター Agent
# 簡潔で効率的な会話を促すプロンプト
#======================================
def generate_summary_system_message(selected_agents):
    """オーケストレーター用システムメッセージ生成"""
    agent_list_str = "、".join(get_agent_names(selected_agents))
    return (
        "あなたは効率的な議論のオーケストレーターです。\n"
        "【重要ルール】\n"
        "1. 余計な挨拶や前置きは省略し、本題に直接入ること\n"
        "2. 各エージェントは1回の発言で完結させること\n"
        "3. 同じ内容を繰り返さないこと\n"
        "4. データ取得後は即座に結論を出すこと\n\n"
        f"参加エージェント: {agent_list_str}\n\n"
        "【進行】\n"
        "1. 質問を確認し、該当エージェントを指名\n"
        "2. エージェントがデータ取得・回答\n"
        "3. 回答をまとめて「以上です。」で終了\n\n"
        "無駄な確認や相談は不要。直接回答を求めてください。"
    )

#======================================
# Dr.Sum Agent Definitions
#======================================
# Load Dr.Sum Agent Definitions from JSON file
def load_dr_sum_agents():
    json_path = os.path.join(os.path.dirname(__file__), 'drsum_data_definitions.json')
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading agent definitions: {e}")
        return []

DR_SUM_AGENTS_DEF = load_dr_sum_agents()

def get_dr_sum_agent_full_name(agent_def):
    """Generate the full agent name like 'Name(ID)' or just 'Name' if ID is the name"""
    if agent_def["id"] == agent_def["name"]:
        return agent_def["name"]
    # Special case for V_在庫状況照会 to match existing behavior if needed, or just follow pattern
    if agent_def["id"] == "V_在庫状況照会":
        return "在庫状況照会"
    return f"{agent_def['name']}({agent_def['id']})"

# Map full agent names to their definitions for easy lookup
DR_SUM_AGENT_MAP = {get_dr_sum_agent_full_name(a): a for a in DR_SUM_AGENTS_DEF}

#======================================
# Agent
# 簡潔で効率的な回答を促すプロンプト
#======================================
def generate_agent_system_message(agent_type, selected_agents):
    """各エージェント用システムメッセージ生成"""
    agent_list_str = "、".join(get_agent_names(selected_agents))
    
    base_msg = (
        f"あなたは{agent_type}の専門家です。\n"
        "【重要ルール】\n"
        "1. 余計な挨拶や前置きは省略すること\n"
        "2. ツールでデータを取得したら、即座に結論を述べること\n"
        "3. 他エージェントへの確認依頼は最小限にすること\n"
        "4. 1回の発言で回答を完結させること\n"
        "5. 同じ内容を繰り返さないこと\n\n"
    )

    if agent_type in DR_SUM_AGENT_MAP:
        agent_def = DR_SUM_AGENT_MAP[agent_type]
        table_id = agent_def["id"]
        base_msg += (
            f"【データソース】データベース「z00_情報系DB」のビュー「{table_id}」\n"
            f"【データ概要】{agent_def['desc']}\n\n"
            "【手順】\n"
            "1. まず `get_schema` でカラム定義を確認\n"
            "2. `execute_select` でデータ取得\n"
            "3. 結果を簡潔にまとめて回答\n\n"
            "推測は禁止。必ずツールでデータを取得してから回答すること。"
        )
    
    return base_msg

#======================================
# グループチャット
#======================================
def create_groupchat(selected_agents, message_callback=None):
    """選択されたエージェントでグループチャットを生成
    
    Args:
        selected_agents: 選択されたエージェント名のリスト
        message_callback: メッセージが生成されたときに呼ばれるコールバック関数
    """
    
    # Base agent map with User and Orchestrator
    agent_map = {
        "オーケストレーター": lambda: AssistantAgent(
            name="オーケストレーター",
            system_message=generate_summary_system_message(selected_agents),
            llm_config=llm_config
        ),
        "User": lambda: user_proxy,
    }

    # Add Dr.Sum agents dynamically
    for agent_name in DR_SUM_AGENT_MAP.keys():
        # Capture agent_name in closure
        agent_map[agent_name] = lambda name=agent_name: AssistantAgent(
            name=name,
            system_message=generate_agent_system_message(name, selected_agents),
            llm_config=llm_config
        )

    agents = [agent_map[name]() for name in selected_agents if name in agent_map]

    # Register tools for all Dr.Sum agents present in the chat
    user_agent = next((agent for agent in agents if agent.name == "User"), None)
    
    # Register tools for all Dr.Sum agents present in the chat
    user_agent = next((agent for agent in agents if agent.name == "User"), None)
    # Register tools for all Dr.Sum agents present in the chat
    user_agent = next((agent for agent in agents if agent.name == "User"), None)
    
    if user_agent:
        dr_sum_agents = [agent for agent in agents if agent.name in DR_SUM_AGENT_MAP]
        
        if dr_sum_agents:
            from drsum_tools import (
                get_database_list, get_table_list, get_schema, 
                execute_select, measure_select
            )
            from autogen import register_function
            
            # Register tools for all Dr.Sum agents
            for dr_sum_agent in dr_sum_agents:
                # get_database_list
                register_function(
                    get_database_list,
                    caller=dr_sum_agent,
                    executor=user_agent,
                    name="get_database_list",
                    description="Dr.Sumのデータベース一覧を取得します。"
                )
                # get_table_list
                register_function(
                    get_table_list,
                    caller=dr_sum_agent,
                    executor=user_agent,
                    name="get_table_list",
                    description="指定したデータベースのテーブル一覧を取得します。"
                )
                # get_schema
                register_function(
                    get_schema,
                    caller=dr_sum_agent,
                    executor=user_agent,
                    name="get_schema",
                    description="指定したテーブルのスキーマ情報（カラム定義など）を取得します。"
                )
                # execute_select
                register_function(
                    execute_select,
                    caller=dr_sum_agent,
                    executor=user_agent,
                    name="execute_select",
                    description="SQL（SELECT文）を実行してデータを取得します。"
                )
                # measure_select
                register_function(
                    measure_select,
                    caller=dr_sum_agent,
                    executor=user_agent,
                    name="measure_select",
                    description="SQL実行のパフォーマンスを計測します。"
                )

#======================================
# 会話の最大回数の設定
# max_round=15で、最大15回までの会話を設定（効率化のため削減）
#======================================
    return GroupChat(agents=agents,
                    messages=[],
                     max_round=15,     # 15回に削減
                     speaker_selection_method="auto",
                     allow_repeat_speaker=False)