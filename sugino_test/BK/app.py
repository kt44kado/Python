#======================================
# ライブラリ
#======================================
import os
import re
import time
import PyPDF2
import streamlit as st
 
from autogen import GroupChatManager
from dotenv import load_dotenv
from agents import create_groupchat
 
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
# エージェント設定　*カラー
# エージェント名とカラーを変更してください。
#======================================
agent_styles = {
    "User": "background-color:#b0c4d6; color:#1a2634;",
    "オーケストレーター": "background-color:#ffe2b2; color:#5a3e1b;",
    "MotionBoard": "background-color:#d1c4e9; color:#2d254c;",
    "SAP": "background-color:#b7d7c9; color:#1b3a2b;",
    "Dr.Sum": "background-color:#f5e9b5; color:#665c1e;",
    "楽々WFⅡ": "background-color:#f3c1c6; color:#6b2b36;",
    "エージェントE": "background-color:#b3d1e6; color:#1b2c3a;",
    "エージェントF": "background-color:#d7ccc8; color:#3e2723;"
}
 
#======================================
# エージェント設定　*アイコン
# エージェント名とアイコンを変更してください。
#======================================
agent_images = {
    "オーケストレーター": "https://img.icons8.com/fluency/96/administrator-male.png",
    "MotionBoard": "https://img.icons8.com/color/96/000000/a.png",
    "SAP": "https://img.icons8.com/color/96/000000/b.png",
    "Dr.Sum": "https://img.icons8.com/color/96/000000/c.png",
    "楽々WFⅡ": "https://img.icons8.com/color/96/000000/d.png",
    "エージェントE": "https://img.icons8.com/color/96/000000/e.png",
    "エージェントF": "https://img.icons8.com/color/96/000000/f.png",
    "User": "https://img.icons8.com/color/96/000000/user.png"
}
 
#======================================
# streamlitの設定
#======================================
def main():
    st.markdown(
#======================================
# 企業ロゴ等があればここにURLを載せてください
#======================================
 
        """
        <div style="text-align:center;">
            <img src="https://logos-world.net/wp-content/uploads/2020/09/Microsoft-Logo.png" width="220"/>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown(
#======================================
# マルチエージェントシステムのタイトルを設定してください
#======================================
        """
        <h2 style="text-align:center;">
            例：マルチエージェントシステム<br> サブタイトル
        </h2>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")
 
#======================================
# エージェント名を変更してください。
#======================================
    selectable_agents = [
        "MotionBoard",
        "SAP",
        "Dr.Sum",
        "楽々WFⅡ",
        "エージェントE",
        "エージェントF"
    ]
 
    theme = st.text_input("テーマを入力してください（必須）", key="discussion_theme")
 
    selected_agents = st.multiselect(
        "エージェント選択",
        options=selectable_agents,
        default=[]
    )
    #オーケストレーターとUserはデフォルト
    selected_agents = ["オーケストレーター"] + selected_agents + ["User"]
 
    # セッション初期化
    if "groupchat" not in st.session_state:
        st.session_state.groupchat = None
    if "manager" not in st.session_state:
        st.session_state.manager = None
    if "user_message" not in st.session_state:
        st.session_state.user_message = None
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "chat_started" not in st.session_state:
        st.session_state.chat_started = False
 
    # PDFファイル読み込み
    pdf_path = st.text_input("参照ファイルパス（任意）", key="pdf_path")
 
    # エージェントへ問い合わせ
    if st.button("エージェントへインプット"):
        user_message = ""
        if pdf_path:
            try:
                user_message = read_pdf(pdf_path)
                st.session_state.user_message = user_message
                st.success("ファイルを読み込みました。")
                st.text_area("読み込んだファイル", user_message, height=200)
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
                return
 
        if theme.strip():
            if user_message:
                user_message += f"\n\n【テーマ】\n{theme.strip()}"
            else:
                user_message = f"【テーマ】\n{theme.strip()}"
 
        if not user_message.strip():
            st.warning("PDFまたはテーマのどちらかを入力してください。")
            return
 
        # グループチャットの初期化
        groupchat = create_groupchat(selected_agents)
        manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)
        st.session_state.groupchat = groupchat
        st.session_state.manager = manager
        st.session_state.initialized = True
        st.session_state.chat_started = False  # チャット開始フラグをリセット
 
        # チャットはここで実行（ターミナルで完了させる）
        groupchat.agents[-1].initiate_chat(manager, message=user_message)
 
    # 会話開始
    if st.session_state.initialized and st.session_state.groupchat:
        if st.button("ディスカッションを開始します"):
            st.session_state.chat_started = True
 
    # 会話履歴の表示
    if st.session_state.initialized and st.session_state.groupchat and st.session_state.chat_started:
        st.markdown("""
        <style>
        .chat-row { margin-bottom: 1.5em; }
        .chat-bubble {
            border-radius: 18px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            padding: 1em 1.2em;
            max-width: 60vw;
            font-size: 1.08em;
            line-height: 1.7;
            position: relative;
            word-break: break-word;
        }
        .chat-label {
            font-weight: bold;
            font-size: 0.98em;
            color: #333;
            margin-bottom: 0.3em;
            display: block;
        }
        .chat-row.left .chat-bubble {
            margin-left: 0.5em;
            background: #f9fafb;
        }
        .chat-row.right .chat-bubble {
            margin-right: 0.5em;
            background: #e3f2fd;
        }
        .chat-row img {
            border-radius: 50%;
            border: 2px solid #fff;
            box-shadow: 0 1px 4px rgba(0,0,0,0.07);
        }
        </style>
        """, unsafe_allow_html=True)
 
        st.markdown("### 各エージェントの会話履歴")
        chat_container = st.container()
 
        with chat_container:
            for idx, msg in enumerate(st.session_state.groupchat.messages):
                if not isinstance(msg, dict):
                    continue
                if msg.get("type") in ["tool_use", "tool_result"]:
                    continue
                name = msg.get("name", "Unknown")
                content = msg.get("content", "")
                if (
                    content is None
                    or (isinstance(content, str) and content.strip() in ["", "None"])
                    or isinstance(content, (list, dict))
                ):
                    continue
                if not isinstance(content, str):
                    content = str(content)
                if (
                    re.search(r"\*{5} (Suggested tool call|Response from calling tool)", content)
                    or content.strip().startswith("***** Suggested tool call")
                    or content.strip().startswith("***** Response from calling tool")
                ):
                    continue
                if (
                    (isinstance(content, str) and (
                        (content.strip().startswith("[{") and content.strip().endswith("}]")) or
                        (content.strip().startswith("{") and content.strip().endswith("}"))
                    ))
                ):
                    continue
                if name == "User" and not content.strip():
                    continue
 
                html_content = "<br>".join(line if line.strip() else "<br>" for line in content.splitlines())
                style = agent_styles.get(name, "background-color:#eeeeee; color:#222;")
                align_class = "left" if idx % 2 == 0 else "right"
 
                if align_class == "left":
                    st.markdown(
                        f"""
                        <div class="chat-row left" style="display:flex;align-items:flex-start;">
                            <img src="{agent_images.get(name, '')}" alt="{name}" width="48" height="48" style="margin-right:1em;align-self:flex-start;"/>
                            <div class="chat-bubble" style="{style}">
                                <span class="chat-label">{name}</span>
                                <span>{html_content}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="chat-row right" style="display:flex;flex-direction:row-reverse;align-items:flex-start;">
                            <img src="{agent_images.get(name, '')}" alt="{name}" width="48" height="48" style="margin-left:1em;align-self:flex-start;"/>
                            <div class="chat-bubble" style="{style}">
                                <span class="chat-label">{name}</span>
                                <span>{html_content}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                time.sleep(5)  # 各メッセージの表示間隔を5秒に設定
 
if __name__ == "__main__":
    main()