import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="ボックス相場スクリーナー", layout="wide")
st.title("📦 ボックス相場スクリーナー")
st.caption(
    "ボリンジャーバンドの外に飛び出さず、バンド内で上下を繰り返している"
    "『レンジ相場・損切りになりにくい』銘柄を探します。トレンドが出ている銘柄"
    "（どちらかに抜けていく銘柄）は除外する方向で評価します。"
)

DEFAULT_TICKERS = """1301 1332 1605 1801 1802 1803 1925 1928 1963 2002
2502 2503 2768 2801 2914 3003 3092 3382 3402 3407
4063 4188 4452 4502 4503 4507 4519 4523 4568 4578
4689 4755 4901 4911 5020 5108 5201 5401 5711 5713
5721 5801 5802 5803 6098 6146 6178 6301 6305 6326
6367 6501 6503 6504 6506 6645 6701 6702 6723 6752
6758 6762 6857 6861 6902 6920 6954 6971 6981 7011
7201 7203 7267 7269 7270 7731 7733 7735 7741 7751
7832 7974 8001 8002 8031 8035 8058 8233 8306 8316
8411 8591 8601 8604 8725 8766 8801 8802 8830 9020
9021 9022 9101 9104 9107 9432 9433 9434 9613 9984""".split()

LOOKBACK = 90  # 判定に使う日数(約4か月)


@st.cache_data(ttl=3600, show_spinner=False)
def get_data(ticker):
    t = str(ticker).strip()
    if not t:
        return None
    symbol = t if t.endswith(".T") else t + ".T"
    try:
        df = yf.download(symbol, period="8mo", interval="1d",
                          auto_adjust=False, progress=False, threads=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[cols].copy().dropna()
        if len(df) < LOOKBACK + 25:
            return None
        return df
    except Exception:
        return None


def bollinger(close, n=20, k=2):
    ma = close.rolling(n).mean()
    std = close.rolling(n).std()
    upper = ma + k * std
    lower = ma - k * std
    pct_b = (close - lower) / (upper - lower)
    return ma, upper, lower, pct_b


def score_stock(df):
    c = df["Close"].astype(float)
    v = df["Volume"].astype(float)

    ma, upper, lower, pct_b = bollinger(c)
    valid = pct_b.dropna().iloc[-LOOKBACK:]
    if len(valid) < LOOKBACK * 0.8:
        return None

    score = 0
    reasons = []
    last = float(c.iloc[-1])

    # --- 1. バンド内滞在率(飛び出した日がどれだけ少ないか) ---
    contain_ratio = float(((valid >= -0.05) & (valid <= 1.05)).mean() * 100)
    if contain_ratio >= 95:
        score += 30; reasons.append(f"バンド内滞在率{contain_ratio:.0f}%(非常に安定)")
    elif contain_ratio >= 85:
        score += 15; reasons.append(f"バンド内滞在率{contain_ratio:.0f}%")
    else:
        score -= 10; reasons.append(f"バンド逸脱が目立つ(滞在率{contain_ratio:.0f}%)")

    # --- 2. 中心線(20日移動平均)がトレンドレスか ---
    ma_valid = ma.dropna()
    if len(ma_valid) >= LOOKBACK:
        ma_chg = float((ma_valid.iloc[-1] / ma_valid.iloc[-LOOKBACK] - 1) * 100)
    else:
        ma_chg = float((ma_valid.iloc[-1] / ma_valid.iloc[0] - 1) * 100)
    if abs(ma_chg) < 5:
        score += 20; reasons.append(f"中心線ほぼ横ばい({ma_chg:+.1f}%)トレンドレス")
    elif abs(ma_chg) < 10:
        score += 8; reasons.append(f"中心線やや傾き({ma_chg:+.1f}%)")
    else:
        score -= 15; reasons.append(f"中心線が大きく傾いている({ma_chg:+.1f}%)トレンド相場の可能性")

    # --- 3. 上限・下限、両方に触れて往復しているか ---
    upper_touches = int((valid >= 0.85).sum())
    lower_touches = int((valid <= 0.15).sum())
    if upper_touches >= 3 and lower_touches >= 3:
        score += 20; reasons.append(f"上限{upper_touches}回・下限{lower_touches}回タッチ(往復レンジ)")
    elif upper_touches >= 2 and lower_touches >= 2:
        score += 10; reasons.append(f"上限{upper_touches}回・下限{lower_touches}回タッチ")
    else:
        score -= 5; reasons.append(f"上下限タッチが少ない(上{upper_touches}回/下{lower_touches}回・偏りの可能性)")

    # --- 4. バンド幅が適度か(狭すぎず広すぎず) ---
    bandwidth_series = ((upper - lower) / ma).dropna().iloc[-LOOKBACK:] * 100
    bandwidth = float(bandwidth_series.mean())
    if 8 <= bandwidth <= 25:
        score += 15; reasons.append(f"バンド幅平均{bandwidth:.1f}%(値幅が適度)")
    elif bandwidth < 8:
        score -= 5; reasons.append(f"バンド幅平均{bandwidth:.1f}%(値幅が狭く利益を取りにくい)")
    else:
        score -= 5; reasons.append(f"バンド幅平均{bandwidth:.1f}%(値幅が広くリスク大)")

    # --- 5. 流動性フィルタ ---
    avgvol20 = float(v.rolling(20).mean().iloc[-1])
    if avgvol20 < 100_000:
        score -= 15; reasons.append("出来高水準が低い(流動性懸念)")

    current_pctb = float(valid.iloc[-1])
    position_note = ""
    if current_pctb <= 0.25:
        position_note = "現在バンド下限付近(押し目候補)"
    elif current_pctb >= 0.75:
        position_note = "現在バンド上限付近(利確ゾーン候補)"
    else:
        position_note = "現在バンド中央付近"

    return {
        "score": max(0, score),
        "last": last,
        "contain_ratio": contain_ratio,
        "ma_chg": ma_chg,
        "upper_touches": upper_touches,
        "lower_touches": lower_touches,
        "bandwidth": bandwidth,
        "current_pctb": current_pctb,
        "position_note": position_note,
        "reasons": ", ".join(reasons),
    }


st.sidebar.header("設定")
text = st.sidebar.text_area("銘柄コード（空白・改行区切り）", " ".join(DEFAULT_TICKERS), height=280)
top_n = st.sidebar.slider("表示件数", 5, 50, 20)
run = st.sidebar.button("🔎 スクリーニング開始", type="primary")

if run:
    tickers = list(dict.fromkeys(text.replace(",", " ").split()))
    results = []
    progress = st.progress(0)
    status = st.empty()

    for i, ticker in enumerate(tickers):
        status.write(f"取得中: {ticker}  ({i + 1}/{len(tickers)})")
        df = get_data(ticker)
        if df is not None:
            try:
                r = score_stock(df)
                if r is not None:
                    results.append({
                        "コード": ticker.replace(".T", ""),
                        "株価": round(r["last"], 1),
                        "スコア": r["score"],
                        "バンド内滞在率%": round(r["contain_ratio"], 1),
                        "現在%B": round(r["current_pctb"], 2),
                        "現在位置": r["position_note"],
                        "上限タッチ": r["upper_touches"],
                        "下限タッチ": r["lower_touches"],
                        "中心線変化率%": round(r["ma_chg"], 1),
                        "バンド幅%": round(r["bandwidth"], 1),
                        "判定理由": r["reasons"]
                    })
            except Exception:
                pass
        progress.progress((i + 1) / len(tickers))

    status.write("完了")
    out = pd.DataFrame(results)
    if out.empty:
        st.error("データを取得できませんでした。ネット接続やYahoo Finance側の制限を確認してください。")
    else:
        out = out.sort_values("スコア", ascending=False).head(top_n).reset_index(drop=True)
        st.subheader(f"ボックス相場候補 TOP {len(out)}")
        st.dataframe(out, use_container_width=True, hide_index=True)
        st.download_button("CSVで保存", out.to_csv(index=False).encode("utf-8-sig"),
                            "range_bound_screener.csv", "text/csv")
else:
    st.info("左側の「スクリーニング開始」を押してください。")
    st.markdown("""
### このアプリが見ているもの(約4か月分のデータで判定)

- **バンド内滞在率** : 過去約4か月のうち、ボリンジャーバンドの外に飛び出した日がどれだけ少ないか
- **中心線のトレンドレス度** : 20日移動平均自体がほぼ横ばいか(傾いていればいずれ片方に抜けるトレンド相場の可能性)
- **上限・下限タッチ回数** : バンドの上下どちらにも複数回触れて、実際に「往復」しているか
- **バンド幅** : 狭すぎず広すぎず、利益を狙いつつリスクを抑えやすい値幅か
- **現在位置(%B)** : 今、バンドのどのあたりにいるか(下限付近なら押し目、上限付近なら利確ゾーンの目安)

### 使い方の一例
スコア上位の銘柄をウォッチリストに入れ、「現在位置」が下限付近になったタイミングで押し目買いを検討し、
上限付近に来たら利益確定を検討する、という往復売買のイメージで使います。

### 注意点
- 過去にレンジで推移していたからといって、今後もそうなる保証はありません。決算発表や材料が出れば一気にレンジを抜けることがあります。
- あくまで過去の値動きパターンからの絞り込みであり、投資助言ではありません。
""")
