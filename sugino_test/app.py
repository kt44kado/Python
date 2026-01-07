#======================================
# ライブラリ
#======================================
import os
import re
import time
import json
import threading
import PyPDF2
import streamlit as st
 
from autogen.agentchat.groupchat import GroupChatManager
from dotenv import load_dotenv
from openai import AzureOpenAI
from agents import create_groupchat, DR_SUM_AGENT_MAP
 
#======================================
# 環境変数の読込み
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
# Azure OpenAI クライアント（自動選択用）
#======================================
azure_client = AzureOpenAI(
    api_key=os.getenv("API_KEY"),
    api_version=os.getenv("API_VERSION"),
    azure_endpoint=os.getenv("API_ENDPOINT")
)
 
#======================================
# PDFファイル読み取り
#======================================
def read_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

#======================================
# 自動エージェント選択機能
#======================================
def auto_select_agents(user_question, available_agents):
    """
    ユーザーの質問内容からLLMを使って適切なエージェントを自動選択する
    """
    # エージェント情報を整形
    agent_info = []
    for agent_name in available_agents:
        agent_def = DR_SUM_AGENT_MAP.get(agent_name, {})
        table_id = agent_def.get("id", agent_name)
        desc = agent_def.get("desc", "")
        agent_info.append(f"- {agent_name}: テーブルID={table_id}, 説明={desc}")
    
    agent_list_text = "\n".join(agent_info)
    
    prompt = f"""あなたはエージェント選択アシスタントです。
ユーザーの質問に回答するために適切なエージェントを選択してください。

# 利用可能なエージェント一覧:
{agent_list_text}

# ユーザーの質問:
{user_question}

# 指示:
上記の質問に回答するために必要なエージェントを選択してください。
選択するエージェントは1つ以上、最大3つまでにしてください。
回答は以下のJSON形式で返してください:
{{"selected_agents": ["エージェント名1", "エージェント名2"]}}

必ずJSON形式のみで回答し、説明は不要です。
"""
    
    try:
        response = azure_client.chat.completions.create(
            model=os.getenv("DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": "あなたはエージェント選択アシスタントです。JSON形式でのみ回答してください。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # JSON部分を抽出
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(result_text)
        selected = result.get("selected_agents", [])
        
        # 有効なエージェントのみをフィルタリング
        valid_agents = [a for a in selected if a in available_agents]
        
        if not valid_agents:
            return available_agents[:3]
        
        return valid_agents
        
    except Exception as e:
        st.warning(f"自動選択でエラーが発生しました: {e}\nデフォルトのエージェントを使用します。")
        return available_agents[:3]

#======================================
# メッセージフィルタリング
#======================================
def should_display_message(msg):
    """表示すべきメッセージかどうかを判定"""
    if not isinstance(msg, dict):
        return False
    if msg.get("type") in ["tool_use", "tool_result"]:
        return False
    
    content = msg.get("content", "")
    if content is None:
        return False
    if isinstance(content, (list, dict)):
        return False
    if not isinstance(content, str):
        content = str(content)
    if content.strip() in ["", "None"]:
        return False
    
    # ツール呼び出し関連のメッセージをフィルタ
    if re.search(r"\*{5} (Suggested tool call|Response from calling tool)", content):
        return False
    if content.strip().startswith("***** Suggested tool call"):
        return False
    if content.strip().startswith("***** Response from calling tool"):
        return False
    
    # JSON形式のメッセージをフィルタ
    if (content.strip().startswith("[{") and content.strip().endswith("}]")):
        return False
    if (content.strip().startswith("{") and content.strip().endswith("}")):
        # JSONっぽい場合はスキップ
        try:
            json.loads(content.strip())
            return False
        except:
            pass
    
    name = msg.get("name", "")
    if name == "User" and not content.strip():
        return False
    
    return True
 
#======================================
# エージェント設定　*カラー
#======================================
agent_styles = {
    "User": "background-color:#b0c4d6; color:#1a2634;",
    "オーケストレーター": "background-color:#ffe2b2; color:#5a3e1b;",
}
DEFAULT_AGENT_STYLE = "background-color:#e8e8e8; color:#333333;"
 
#======================================
# エージェント設定　*アイコン
#======================================
agent_images = {
    "オーケストレーター": "https://img.icons8.com/fluency/96/administrator-male.png",
    "User": "https://img.icons8.com/color/96/000000/user.png"
}
DEFAULT_AGENT_IMAGE = "https://img.icons8.com/color/96/000000/robot.png"

#======================================
# メッセージ表示用HTML生成
#======================================
def render_message_html(name, content, idx):
    """メッセージをHTML形式で描画"""
    html_content = "<br>".join(line if line.strip() else "<br>" for line in content.splitlines())
    style = agent_styles.get(name, DEFAULT_AGENT_STYLE)
    align_class = "left" if idx % 2 == 0 else "right"
    agent_image = agent_images.get(name, DEFAULT_AGENT_IMAGE)
    
    if align_class == "left":
        return f"""
        <div class="chat-row left" style="display:flex;align-items:flex-start;margin-bottom:1.5em;">
            <img src="{agent_image}" alt="{name}" width="48" height="48" style="margin-right:1em;align-self:flex-start;border-radius:50%;"/>
            <div style="border-radius:18px;box-shadow:0 2px 8px rgba(0,0,0,0.08);padding:1em 1.2em;max-width:60vw;{style}">
                <span style="font-weight:bold;display:block;margin-bottom:0.3em;">{name}</span>
                <span>{html_content}</span>
            </div>
        </div>
        """
    else:
        return f"""
        <div class="chat-row right" style="display:flex;flex-direction:row-reverse;align-items:flex-start;margin-bottom:1.5em;">
            <img src="{agent_image}" alt="{name}" width="48" height="48" style="margin-left:1em;align-self:flex-start;border-radius:50%;"/>
            <div style="border-radius:18px;box-shadow:0 2px 8px rgba(0,0,0,0.08);padding:1em 1.2em;max-width:60vw;{style}">
                <span style="font-weight:bold;display:block;margin-bottom:0.3em;">{name}</span>
                <span>{html_content}</span>
            </div>
        </div>
        """
 
#======================================
# streamlitの設定
#======================================
def main():
    st.set_page_config(page_title="マルチエージェントシステム", layout="wide")
    
    st.markdown(
        """
        <div style="text-align:center;">
            <img src="https://logos-world.net/wp-content/uploads/2020/09/Microsoft-Logo.png" width="220"/>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown(
        """
        <h2 style="text-align:center;">
            マルチエージェントシステム<br><small>Dr.Sum データ照会</small>
        </h2>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")
 
    selectable_agents = list(DR_SUM_AGENT_MAP.keys())
 
    theme = st.text_input("質問を入力してください", key="discussion_theme", 
                          placeholder="例: ｳｴｽの在庫状況を教えて")

    col1, col2 = st.columns([1, 1])
    with col1:
        auto_mode = st.checkbox("🤖 エージェント自動選択", value=True,
                                help="質問内容から適切なエージェントを自動選択")
    
    if auto_mode:
        st.info("📌 自動選択モード: 質問内容に基づいて適切なエージェントが選択されます")
        selected_agents_manual = []
    else:
        selected_agents_manual = st.multiselect(
            "エージェント選択",
            options=selectable_agents,
            default=[]
        )
 
    # セッション初期化
    if "groupchat" not in st.session_state:
        st.session_state.groupchat = None
    if "manager" not in st.session_state:
        st.session_state.manager = None
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "chat_running" not in st.session_state:
        st.session_state.chat_running = False
    if "displayed_count" not in st.session_state:
        st.session_state.displayed_count = 0
 
    # PDFファイル読み込み（折りたたみ）
    with st.expander("📄 PDFファイルから読み込む（任意）"):
        pdf_path = st.text_input("ファイルパス", key="pdf_path")
 
    # チャット表示エリア
    chat_placeholder = st.empty()

    # エージェントへ問い合わせ
    if st.button("🚀 質問を送信", type="primary"):
        user_message = ""
        if pdf_path:
            try:
                user_message = read_pdf(pdf_path)
                st.success("ファイルを読み込みました。")
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
                return
 
        if theme.strip():
            if user_message:
                user_message += f"\n\n【質問】\n{theme.strip()}"
            else:
                user_message = theme.strip()
 
        if not user_message.strip():
            st.warning("質問を入力してください。")
            return
 
        # エージェント選択
        if auto_mode:
            with st.spinner("🔍 適切なエージェントを選択中..."):
                auto_selected = auto_select_agents(user_message, selectable_agents)
                st.success(f"選択されたエージェント: {', '.join(auto_selected)}")
            selected_agents = ["オーケストレーター"] + auto_selected + ["User"]
        else:
            if not selected_agents_manual:
                st.warning("エージェントを選択してください。")
                return
            selected_agents = ["オーケストレーター"] + selected_agents_manual + ["User"]
 
        # グループチャットの初期化
        groupchat = create_groupchat(selected_agents)
        manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)
        st.session_state.groupchat = groupchat
        st.session_state.manager = manager
        st.session_state.initialized = True
        st.session_state.displayed_count = 0
        st.session_state.chat_running = True
 
        # リアルタイム表示用のプレースホルダー
        status_placeholder = st.empty()
        chat_display = st.container()
        
        status_placeholder.info("🔄 エージェントが回答を生成中...")
        
        # チャット実行（バックグラウンドで実行し、進捗を表示）
        def run_chat():
            groupchat.agents[-1].initiate_chat(manager, message=user_message)
            st.session_state.chat_running = False
        
        # スレッドで実行
        chat_thread = threading.Thread(target=run_chat)
        chat_thread.start()
        
        # リアルタイムで更新
        with chat_display:
            last_count = 0
            while st.session_state.chat_running or last_count < len(groupchat.messages):
                current_messages = groupchat.messages
                
                # 新しいメッセージがあれば表示
                for idx in range(last_count, len(current_messages)):
                    msg = current_messages[idx]
                    if should_display_message(msg):
                        name = msg.get("name", "Unknown")
                        content = msg.get("content", "")
                        if not isinstance(content, str):
                            content = str(content)
                        
                        st.markdown(render_message_html(name, content, idx), unsafe_allow_html=True)
                
                last_count = len(current_messages)
                time.sleep(0.5)  # 0.5秒ごとにチェック
        
        chat_thread.join()
        status_placeholder.success("✅ 回答が完了しました")
        st.session_state.chat_running = False

    # 既存の会話履歴を表示
    if st.session_state.initialized and st.session_state.groupchat and not st.session_state.chat_running:
        st.markdown("---")
        st.markdown("### 💬 会話履歴")
        
        for idx, msg in enumerate(st.session_state.groupchat.messages):
            if should_display_message(msg):
                name = msg.get("name", "Unknown")
                content = msg.get("content", "")
                if not isinstance(content, str):
                    content = str(content)
                st.markdown(render_message_html(name, content, idx), unsafe_allow_html=True)

        # 追加質問
        st.markdown("---")
        st.markdown("### 📝 追加質問")
        
        with st.form(key="follow_up_form"):
            follow_up_text = st.text_area("追加の質問を入力", height=80)
            submit_button = st.form_submit_button("送信")
            
        if submit_button and follow_up_text.strip():
            st.session_state.groupchat.max_round += 10
            st.session_state.chat_running = True
            
            status = st.empty()
            status.info("🔄 回答を生成中...")
            
            st.session_state.groupchat.agents[-1].initiate_chat(
                st.session_state.manager,
                message=follow_up_text.strip(),
                clear_history=False
            )
            
            st.session_state.chat_running = False
            status.success("✅ 完了")
            st.rerun()
 
if __name__ == "__main__":
    main()