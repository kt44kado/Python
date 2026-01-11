# 入出力Box
import streamlit as st

st.title("テキスト出力アプリ")

# 入力用のテキストボックス
user_input = st.text_input("ここに入力してください")

# Runボタン
if st.button("Run"):
    # 出力用のテキストエリア（入力された文字を値として渡す）
    st.text_area("出力結果", value=user_input, height=100)