# 出力の履歴表示
import streamlit as st

st.title("テキスト追記アプリ")

# 1. 蓄積するテキストを保存する変数を初期化（初回のみ実行）
if "history" not in st.session_state:
    st.session_state.history = ""

# 入力欄
user_input = st.text_input("追加したいテキストを入力してください")

# Runボタン
if st.button("Run"):
    if user_input:
        # 2. 現在の履歴に新しい入力と改行を追加して保存
        st.session_state.history += user_input + "\n"
        # 入力後にテキストボックスを空にしたい場合は工夫が必要ですが、
        # 基本はこの形で履歴が更新されます。

# 3. 出力ボックス（st.text_area）
# heightを指定することで、これを超えるとスクロールバーが出ます。
st.text_area(
    label="出力結果（履歴）", 
    value=st.session_state.history, 
    height=200,  # 高さを200ピクセルに固定
    disabled=True # 編集不可にする場合（任意）
)