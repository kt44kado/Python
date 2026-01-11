import streamlit as st


# サイドバー
st.sidebar.title("コントロール")
name = st.sidebar.text_input("お名前", value="ゲスト")
level = st.sidebar.slider("難易度", 1, 10, 3)
agree = st.sidebar.checkbox("規約に同意する", value=True)

col1, col2 = st.columns(2)
with col1:
    st.write(f"こんにちは、**{name}** さん！")
with col2:
    if agree:
        st.success(f"難易度は {level} に設定されました。")
    else:
        st.warning("同意が必要です。")

# セレクトボックス & ボタン
option = st.selectbox("好きなグラフ", ["折れ線", "棒", "散布図"])
if st.button("送信"):
    st.info(f"{option} が選択されました")
