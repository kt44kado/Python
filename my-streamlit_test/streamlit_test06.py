import streamlit as st

if "logs" not in st.session_state:
    st.session_state.logs = []

# 入力確定時に呼ばれる関数
def handle_input():
    user_val = st.session_state.current_input
    if user_val:
        st.session_state.logs.append(f"> {user_val}")
        # ここで処理を実行
        st.session_state.logs.append(f"Output: {user_val.upper()}")
        # 入力欄を空にする
        st.session_state.current_input = ""

# 履歴の表示
for log in st.session_state.logs:
    st.text(log)

# 入力欄
st.text_input("Input:", key="current_input", on_change=handle_input)