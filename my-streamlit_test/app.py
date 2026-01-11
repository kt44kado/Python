# app.py
import time
import numpy as np
import pandas as pd
import streamlit as st

# ===== 基本設定 =====
st.set_page_config(page_title="売上ダッシュボード（ダミー）", page_icon="📈", layout="wide")

# ===== データ生成（ダミー） =====
@st.cache_data
def make_data(seed: int = 42, days: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days)
    cats = ["A_雑貨", "B_食品", "C_家電", "D_アパレル"]
    rows = []
    for d in dates:
        for c in cats:
            base = {"A_雑貨": 12000, "B_食品": 15000, "C_家電": 20000, "D_アパレル": 17000}[c]
            season = 1.0 + 0.2*np.sin(2*np.pi*(d.dayofyear)/365)
            trend = 1.0 + (d - dates[0]).days/365*0.3
            noise = rng.normal(0, 2500)
            sales = max(0, base*season*trend + noise)
            qty = max(0, int(sales / rng.uniform(800, 2500)))
            rows.append([d, c, sales, qty])
    df = pd.DataFrame(rows, columns=["date", "category", "sales", "qty"])
    df["date"] = pd.to_datetime(df["date"])
    return df

df = make_data()

# ===== サイドバー（フィルタ） =====
st.sidebar.header("フィルタ")
min_date, max_date = df["date"].min(), df["date"].max()
date_range = st.sidebar.date_input("期間", (max_date - pd.Timedelta(days=60), max_date),
                                   min_value=min_date, max_value=max_date)
selected_cats = st.sidebar.multiselect("カテゴリ", sorted(df["category"].unique()),
                                       default=["A_雑貨", "B_食品"])
smooth = st.sidebar.slider("移動平均（日）", 1, 30, 7)
download = st.sidebar.checkbox("CSVをダウンロード用に整形", value=True)

# ===== データ整形 =====
start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[-1])
mask = (df["date"] >= start) & (df["date"] <= end) & (df["category"].isin(selected_cats))
view = df.loc[mask].copy()

# KPI（期間トータル）
total_sales = view["sales"].sum()
total_qty = view["qty"].sum()
avg_order = total_sales / total_qty if total_qty else 0

# 前期間（同日数）との比較
days = (end - start).days + 1
prev_start = start - pd.Timedelta(days=days)
prev_end = start - pd.Timedelta(days=1)
prev_mask = (df["date"] >= prev_start) & (df["date"] <= prev_end) & (df["category"].isin(selected_cats))
prev = df.loc[prev_mask]
prev_sales = prev["sales"].sum()

delta_sales = ((total_sales - prev_sales) / prev_sales * 100) if prev_sales else np.nan

# ===== レイアウト =====
st.title("📈 売上ダッシュボード（ダミーデータ）")
st.caption(f"期間: {start.date()} 〜 {end.date()} / カテゴリ: {', '.join(selected_cats) or '未選択'}")

k1, k2, k3 = st.columns(3)
k1.metric("売上合計（円）", f"{int(total_sales):,}", None if np.isnan(delta_sales) else f"{delta_sales:.1f}%")
k2.metric("販売数量（個）", f"{int(total_qty):,}")
k3.metric("平均単価（円）", f"{int(avg_order):,}")

# 売上推移（移動平均）
daily = (view.groupby("date", as_index=False)[["sales", "qty"]].sum()
              .sort_values("date"))
if smooth > 1 and not daily.empty:
    daily["sales_smooth"] = daily["sales"].rolling(smooth, min_periods=1).mean()
else:
    daily["sales_smooth"] = daily["sales"]

c1, c2 = st.columns([2, 1])
with c1:
    st.subheader("日次売上推移")
    st.line_chart(daily.set_index("date")[["sales", "sales_smooth"]])

with c2:
    st.subheader("カテゴリ別売上（合計）")
    cat_sales = view.groupby("category", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
    st.bar_chart(cat_sales.set_index("category"))

st.subheader("明細（期間・カテゴリで抽出）")
st.dataframe(view.sort_values(["date", "category"]).reset_index(drop=True), use_container_width=True)

# ダウンロード
if download:
    csv = view.to_csv(index=False).encode("utf-8-sig")
    st.download_button("CSVダウンロード（UTF-8 BOM）", csv, file_name="sales_filtered.csv", mime="text/csv")

# 処理の見える化（体験用）
with st.expander("処理サンプル（スピナー表示の例）"):
    with st.spinner("集計中..."):
        time.sleep(0.5)
    st.success("完了！")

st.caption("Tip: pages/ ディレクトリを作るとマルチページ化できます。")