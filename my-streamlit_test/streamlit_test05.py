# 簡易ターミナル風アプリ
import streamlit as st

st.title("簡易ターミナル風アプリ")

# 1. セッション状態で履歴を保持する（初回のみ初期化）
if "history" not in st.session_state:
    st.session_state.history = []

# 2. 過去の履歴をすべて表示する
for chat in st.session_state.history:
    with st.chat_message(chat["role"]):
        st.write(chat["content"])

# 3. 入力ボックス（Enterで送信される）
if prompt := st.chat_input("コマンドを入力してください..."):
    
    # ユーザーの入力を履歴に追加
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 4. 実行・出力処理（ここに好きなPython処理を書く）
    response = f"実行結果: {prompt} を受け取りました"
    
    # 実行結果を履歴に追加
    st.session_state.history.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)