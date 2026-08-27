import os
import json
import math
import time
import re
import warnings
from datetime import datetime, timedelta, timezone

import streamlit as st
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import ta
import yfinance as yf
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

warnings.filterwarnings("ignore")
load_dotenv()

# 한국 표준시(KST) 정의 (UTC+9)
KST = timezone(timedelta(hours=9))

def get_current_kst_time_str():
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M")

st.set_page_config(
    page_title="AI Multi-Asset Analyst Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# 📌 모바일 반응형 & 텍스트 넘침 방지 커스텀 CSS 주입
# -------------------------------------------------------------
st.markdown("""
<style>
    html, body, [data-testid="stAppViewContainer"] {
        overflow-x: hidden !important;
        max-width: 100vw !important;
    }
    
    div, p, span, h1, h2, h3, h4, h5, h6 {
        overflow-wrap: break-word !important;
        word-break: break-word !important;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-top: 1.2rem !important;
            padding-left: 0.6rem !important;
            padding-right: 0.6rem !important;
            padding-bottom: 2rem !important;
        }

        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 8px !important;
        }
        
        [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            flex: 1 1 calc(50% - 10px) !important;
            min-width: calc(50% - 10px) !important;
            max-width: 100% !important;
            margin-bottom: 4px !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.15rem !important;
            line-height: 1.25 !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.72rem !important;
            line-height: 1.2 !important;
        }
        [data-testid="stMetricDelta"] {
            font-size: 0.68rem !important;
            line-height: 1.2 !important;
        }
        
        h2 {
            font-size: 1.35rem !important;
,        }
        h4, h5 {
            font-size: 0.95rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 0. 영구 저장소 (history.json) 관리 함수
# -------------------------------------------------------------
HISTORY_FILE = "history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_history(history_data):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# --- 세션 상태 초기화 ---
if "history" not in st.session_state:
    st.session_state.history = load_history()

if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "TSLA"

if "last_analysis_result" not in st.session_state:
    st.session_state.last_analysis_result = None

# -------------------------------------------------------------
# 1. 티커 정규화 함수 (GOLD, BTC 입력 시 야후 티커로 매핑)
# -------------------------------------------------------------
def normalize_ticker(raw_input: str) -> tuple:
    clean = raw_input.strip().upper()
    if clean in ["GOLD", "금", "GC"]:
        return "GC=F", "금 (Gold Futures)"
    elif clean in ["BTC", "비트코인", "BITCOIN"]:
        return "BTC-USD", "비트코인 (Bitcoin)"
    else:
        return clean, f"주식 ({clean})"

# -------------------------------------------------------------
# 2. RAG 데이터 수집 모듈 (기술적 지표, BB Squeeze, VWAP, Volume Profile POC 등)
# -------------------------------------------------------------
def get_stock_info_with_retry(stock, retries=3):
    for attempt in range(retries):
        try:
            info = stock.info
            if isinstance(info, dict) and len(info) > 10 and any(k in info for k in ['marketCap', 'trailingPE', 'forwardPE', 'trailingEps', 'bookValue', 'currentPrice']):
                return info, "stock.info"
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(1.0 * (attempt + 1))
            
    try:
        fallback_info = stock.info or {}
        if fallback_info and len(fallback_info) > 5:
            return fallback_info, "stock.info"
    except Exception:
        pass
    return {}, "stock.fast_info"

def fetch_stock_technical_data(ticker: str):
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")
    if df.empty:
        df = stock.history(period="6mo")
    if df.empty:
        return {}, "N/A", {}, "N/A", "N/A", pd.DataFrame(), {}
    
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_60'] = df['Close'].rolling(window=60).mean()
    df['SMA_120'] = df['Close'].rolling(window=120).mean()
    
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    try:
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    except Exception:
        df['ATR'] = None

    try:
        df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window=14)
    except Exception:
        df['MFI'] = None

    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['BB_Mid'] = bb.bollinger_mavg()
    df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['BB_Mid'] * 100

    if df['ATR'] is not None:
        df['KC_High'] = df['SMA_20'] + (1.5 * df['ATR'])
        df['KC_Low'] = df['SMA_20'] - (1.5 * df['ATR'])
        df['BB_Squeeze_On'] = (df['BB_High'] < df['KC_High']) & (df['BB_Low'] > df['KC_Low'])
    else:
        df['BB_Squeeze_On'] = False

    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['TP_Vol'] = df['Typical_Price'] * df['Volume']
    df['Cumulative_VWAP'] = df['TP_Vol'].cumsum() / df['Volume'].cumsum()
    df['Rolling_VWAP_20'] = df['TP_Vol'].rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else latest
    last_date = df.index[-1].strftime("%Y-%m-%d")
    
    high_52w_calc = round(float(df['High'].max()), 2)
    low_52w_calc = round(float(df['Low'].min()), 2)
    
    df_6m = df.tail(126) if len(df) >= 126 else df
    high_6m = float(df_6m['High'].max())
    low_6m = float(df_6m['Low'].min())
    diff_hl = high_6m - low_6m
    
    fibonacci_levels = {
        "high_6m": round(high_6m, 2),
        "low_6m": round(low_6m, 2),
        "fib_23.6%": round(high_6m - (0.236 * diff_hl), 2),
        "fib_38.2%": round(high_6m - (0.382 * diff_hl), 2),
        "fib_50.0%": round(high_6m - (0.500 * diff_hl), 2),
        "fib_61.8%": round(high_6m - (0.618 * diff_hl), 2)
    }
    
    volume_profile = {}
    try:
        num_bins = 30
        price_bins = np.linspace(low_6m, high_6m, num_bins + 1)
        bin_indices = np.digitize(df_6m['Close'], price_bins) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)
        
        vol_by_bin = np.zeros(num_bins)
        for idx_val, vol_val in zip(bin_indices, df_6m['Volume']):
            vol_by_bin[idx_val] += vol_val
            
        poc_idx = int(np.argmax(vol_by_bin))
        poc_price = round(float((price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2), 2)
        
        tot_vol = np.sum(vol_by_bin)
        target_va_vol = tot_vol * 0.70
        sorted_indices = np.argsort(vol_by_bin)[::-1]
        cum_va = 0.0
        va_bins = []
        for s_idx in sorted_indices:
            cum_va += vol_by_bin[s_idx]
            va_bins.append(s_idx)
            if cum_va >= target_va_vol:
                break
        min_va_bin = min(va_bins)
        max_va_bin = max(va_bins)
        val_price = round(float(price_bins[min_va_bin]), 2)
        vah_price = round(float(price_bins[max_va_bin + 1]), 2)
        
        volume_profile = {
            "poc_price": poc_price,
            "vah_price": vah_price,
            "val_price": val_price,
            "value_area_range": f"${val_price} ~ ${vah_price}"
        }
    except Exception:
        volume_profile = {
            "poc_price": "N/A",
            "vah_price": "N/A",
            "val_price": "N/A",
            "value_area_range": "N/A"
        }
    
    atr_val = round(float(latest['ATR']), 2) if pd.notnull(latest['ATR']) else "N/A"
    
    is_sqz_now = bool(latest['BB_Squeeze_On'])
    is_sqz_prev = bool(prev['BB_Squeeze_On'])
    if is_sqz_now:
        squeeze_status = "⚠️ 스퀴즈 진행 중 (에너지 응축/변동성 폭발 임박)"
    elif is_sqz_prev and not is_sqz_now:
        if latest['Close'] > latest['BB_Mid']:
            squeeze_status = "🚀 상방 스퀴즈 분출 (상승 랠리 가속)"
        else:
            squeeze_status = "📉 하방 스퀴즈 이탈 (하락 가속 경보)"
    else:
        squeeze_status = "정상 변동성 구간 (스퀴즈 해제 상태)"

    data = {
        "current_price": round(float(latest['Close']), 2),
        "atr_14": atr_val,
        "atr_stop_1_5x": round(float(latest['Close']) - (1.5 * float(latest['ATR'])), 2) if pd.notnull(latest['ATR']) else "N/A",
        "atr_stop_2_0x": round(float(latest['Close']) - (2.0 * float(latest['ATR'])), 2) if pd.notnull(latest['ATR']) else "N/A",
        "sma_20": round(float(latest['SMA_20']), 2) if pd.notnull(latest['SMA_20']) else "N/A",
        "sma_60": round(float(latest['SMA_60']), 2) if pd.notnull(latest['SMA_60']) else "N/A",
        "sma_120": round(float(latest['SMA_120']), 2) if pd.notnull(latest['SMA_120']) else "N/A",
        "rsi_14": round(float(latest['RSI']), 2) if pd.notnull(latest['RSI']) else "N/A",
        "mfi_14": round(float(latest['MFI']), 2) if pd.notnull(latest['MFI']) else "N/A",
        "macd": round(float(latest['MACD']), 2) if pd.notnull(latest['MACD']) else "N/A",
        "macd_signal": round(float(latest['MACD_Signal']), 2) if pd.notnull(latest['MACD_Signal']) else "N/A",
        "macd_hist": round(float(latest['MACD_Hist']), 2) if pd.notnull(latest['MACD_Hist']) else "N/A",
        "bb_upper": round(float(latest['BB_High']), 2) if pd.notnull(latest['BB_High']) else "N/A",
        "bb_middle": round(float(latest['BB_Mid']), 2) if pd.notnull(latest['BB_Mid']) else "N/A",
        "bb_lower": round(float(latest['BB_Low']), 2) if pd.notnull(latest['BB_Low']) else "N/A",
        "bb_width_pct": round(float(latest['BB_Width']), 2) if pd.notnull(latest['BB_Width']) else "N/A",
        "bb_squeeze_status": squeeze_status,
        "vwap_1y": round(float(latest['Cumulative_VWAP']), 2) if pd.notnull(latest['Cumulative_VWAP']) else "N/A",
        "vwap_20d": round(float(latest['Rolling_VWAP_20']), 2) if pd.notnull(latest['Rolling_VWAP_20']) else "N/A",
        "poc_price_6m": volume_profile.get("poc_price", "N/A"),
        "value_area_range_6m": volume_profile.get("value_area_range", "N/A")
    }
    return data, last_date, fibonacci_levels, high_52w_calc, low_52w_calc, df, volume_profile

def run_strategy_backtest(df: pd.DataFrame):
    if df is None or len(df) < 60:
        return None

    b_df = df.copy().dropna(subset=['Close', 'SMA_20', 'MACD', 'MACD_Signal', 'ATR', 'Cumulative_VWAP'])
    if len(b_df) < 30:
        return None

    bh_return = (b_df['Close'].iloc[-1] - b_df['Close'].iloc[0]) / b_df['Close'].iloc[0] * 100

    pos1 = 0
    entry_p1 = 0
    trades1 = []

    for i in range(1, len(b_df)):
        cur = b_df.iloc[i]
        prev = b_df.iloc[i-1]
        
        if pos1 == 1:
            stop_price = entry_p1 - (1.5 * cur['ATR']) if pd.notnull(cur['ATR']) else entry_p1 * 0.93
            if cur['Close'] < stop_price or cur['MACD_Hist'] < 0:
                ret = (cur['Close'] - entry_p1) / entry_p1
                trades1.append(ret)
                pos1 = 0
                entry_p1 = 0
        
        if pos1 == 0:
            cond_macd = (prev['MACD_Hist'] <= 0 and cur['MACD_Hist'] > 0)
            cond_trend = cur['Close'] > cur['SMA_20']
            cond_vwap = cur['Close'] > cur['Rolling_VWAP_20'] if pd.notnull(cur['Rolling_VWAP_20']) else True
            if cond_macd and cond_trend and cond_vwap:
                pos1 = 1
                entry_p1 = cur['Close']

    if pos1 == 1:
        ret = (b_df['Close'].iloc[-1] - entry_p1) / entry_p1
        trades1.append(ret)

    pos2 = 0
    entry_p2 = 0
    trades2 = []

    for i in range(1, len(b_df)):
        cur = b_df.iloc[i]
        
        if pos2 == 1:
            if cur['Close'] >= cur['Cumulative_VWAP'] or cur['RSI'] >= 65 or cur['Close'] < (cur['BB_Low'] * 0.97):
                ret = (cur['Close'] - entry_p2) / entry_p2
                trades2.append(ret)
                pos2 = 0
                entry_p2 = 0
        
        if pos2 == 0:
            cond_val = cur['Close'] < cur['Cumulative_VWAP']
            cond_rsi = cur['RSI'] < 42
            cond_bb = cur['Close'] > cur['BB_Low']
            if cond_val and cond_rsi and cond_bb:
                pos2 = 1
                entry_p2 = cur['Close']

    if pos2 == 1:
        ret = (b_df['Close'].iloc[-1] - entry_p2) / entry_p2
        trades2.append(ret)

    def calc_stats(trades):
        if not trades:
            return {"total_ret": 0.0, "win_rate": 0.0, "trades_count": 0, "profit_factor": 0.0, "mdd": 0.0}
        
        cum = 1.0
        peak = 1.0
        mdd = 0.0
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t < 0]
        
        for t in trades:
            cum *= (1.0 + t)
            if cum > peak:
                peak = cum
            dd = (peak - cum) / peak
            if dd > mdd:
                mdd = dd
                
        tot_ret = (cum - 1.0) * 100
        win_rate = (len(wins) / len(trades)) * 100 if trades else 0
        sum_win = sum(wins) if wins else 0
        sum_loss = abs(sum(losses)) if losses else 0
        pf = round(sum_win / sum_loss, 2) if sum_loss > 0 else (99.9 if sum_win > 0 else 0.0)

        return {
            "total_ret": round(tot_ret, 2),
            "win_rate": round(win_rate, 1),
            "trades_count": len(trades),
            "profit_factor": pf,
            "mdd": round(mdd * 100, 2)
        }

    return {
        "benchmark_buy_and_hold": round(bh_return, 2),
        "strategy_1_momentum_squeeze": calc_stats(trades1),
        "strategy_2_vwap_mean_reversion": calc_stats(trades2)
    }

def fetch_nearest_options_data(ticker: str, retries: int = 3):
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            expirations = stock.options
            if not expirations:
                if attempt < retries - 1:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                return None
            
            nearest_exp = expirations[0]
            opt_chain = stock.option_chain(nearest_exp)
            calls = opt_chain.calls
            puts = opt_chain.puts
            
            if calls.empty or puts.empty:
                if attempt < retries - 1:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                return None
                
            call_max_oi_row = calls.loc[calls['openInterest'].idxmax()] if calls['openInterest'].notnull().any() and calls['openInterest'].max() > 0 else calls.iloc[0]
            call_max_vol_row = calls.loc[calls['volume'].idxmax()] if calls['volume'].notnull().any() and calls['volume'].max() > 0 else calls.iloc[0]
            
            put_max_oi_row = puts.loc[puts['openInterest'].idxmax()] if puts['openInterest'].notnull().any() and puts['openInterest'].max() > 0 else puts.iloc[0]
            put_max_vol_row = puts.loc[puts['volume'].idxmax()] if puts['volume'].notnull().any() and puts['volume'].max() > 0 else puts.iloc[0]
            
            tot_call_vol = calls['volume'].sum() if calls['volume'].notnull().any() else 0
            tot_put_vol = puts['volume'].sum() if puts['volume'].notnull().any() else 0
            pc_ratio = round(tot_put_vol / tot_call_vol, 2) if tot_call_vol > 0 else "N/A"

            return {
                "expiration_date": nearest_exp,
                "pc_volume_ratio": pc_ratio,
                "call_max_oi": {
                    "strike": call_max_oi_row.get("strike", "N/A"),
                    "oi": int(call_max_oi_row.get("openInterest", 0)) if pd.notnull(call_max_oi_row.get("openInterest")) else 0,
                    "price": round(float(call_max_oi_row.get("lastPrice", 0)), 2)
                },
                "call_max_vol": {
                    "strike": call_max_vol_row.get("strike", "N/A"),
                    "volume": int(call_max_vol_row.get("volume", 0)) if pd.notnull(call_max_vol_row.get("volume")) else 0,
                    "price": round(float(call_max_vol_row.get("lastPrice", 0)), 2)
                },
                "put_max_oi": {
                    "strike": put_max_oi_row.get("strike", "N/A"),
                    "oi": int(put_max_oi_row.get("openInterest", 0)) if pd.notnull(put_max_oi_row.get("openInterest")) else 0,
                    "price": round(float(put_max_oi_row.get("lastPrice", 0)), 2)
                },
                "put_max_vol": {
                    "strike": put_max_vol_row.get("strike", "N/A"),
                    "volume": int(put_max_vol_row.get("volume", 0)) if pd.notnull(put_max_vol_row.get("volume")) else 0,
                    "price": round(float(put_max_vol_row.get("lastPrice", 0)), 2)
                }
            }
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.0 * (attempt + 1))
                continue
            return None
    return None

def fetch_macro_indicators():
    macro_data = {}
    try:
        end = datetime.now()
        start = end - timedelta(days=30)
        fred_res = web.DataReader('DGS10', 'fred', start, end).dropna()
        dgs10 = fred_res.iloc[-1, 0]
        macro_data["us_10y_yield"] = {
            "source": "FRED (Federal Reserve Economic Data)",
            "value": f"{round(float(dgs10), 2)}%",
            "date": fred_res.index[-1].strftime("%Y-%m-%d")
        }
    except Exception:
        macro_data["us_10y_yield"] = {"source": "FRED", "value": "N/A", "date": "N/A"}
        
    asset_tickers = [
        ("vix", "^VIX", "CBOE Volatility Index"),
        ("dollar_index", "DX-Y.NYB", "ICE US Dollar Index"),
        ("wti_oil", "CL=F", "NYMEX WTI Crude Oil"),
        ("gold", "GC=F", "COMEX Gold Futures"),
        ("bitcoin", "BTC-USD", "Binance/Coinbase Crypto Market")
    ]
    for name, ticker, src_name in asset_tickers:
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if not hist.empty:
                macro_data[name] = {
                    "source": src_name,
                    "value": round(float(hist['Close'].iloc[-1]), 2),
                    "date": hist.index[-1].strftime("%Y-%m-%d")
                }
            else:
                macro_data[name] = {"source": src_name, "value": "N/A", "date": "N/A"}
        except Exception:
            macro_data[name] = {"source": src_name, "value": "N/A", "date": "N/A"}
    return macro_data

def format_market_cap(market_cap):
    if not market_cap or market_cap == "N/A":
        return "N/A"
    try:
        mc = float(market_cap)
        if mc >= 1e12:
            return f"${mc / 1e12:.2f}T"
        elif mc >= 1e9:
            return f"${mc / 1e9:.2f}B"
        elif mc >= 1e6:
            return f"${mc / 1e6:.2f}M"
        return f"${mc:,.0f}"
    except Exception:
        return str(market_cap)

def fetch_hedge_funds_and_short_intel(stock, info):
    intel = {
        "top_holders": [],
        "short_intel": {}
    }
    try:
        inst_df = stock.institutional_holders
        if inst_df is not None and not inst_df.empty:
            for _, row in inst_df.head(6).iterrows():
                holder_name = str(row.get("Holder", "N/A")).strip()
                shares_val = row.get("Shares", 0)
                pct_out_val = row.get("% Out", 0)
                val_val = row.get("Value", 0)
                pct_str = f"{pct_out_val * 100:.2f}%" if pd.notnull(pct_out_val) and pct_out_val < 1.0 else f"{pct_out_val:.2f}%"
                val_str = f"${val_val / 1e9:.2f}B" if pd.notnull(val_val) and val_val >= 1e9 else (f"${val_val / 1e6:.1f}M" if pd.notnull(val_val) and val_val >= 1e6 else "-")
                intel["top_holders"].append({
                    "holder": holder_name,
                    "shares": f"{int(shares_val):,}" if pd.notnull(shares_val) else "-",
                    "percent_out": pct_str,
                    "value": val_str
                })
    except Exception:
        pass

    try:
        short_float = info.get("shortPercentOfFloat", None)
        short_ratio = info.get("shortRatio", None)
        shares_short = info.get("sharesShort", None)
        shares_short_prior = info.get("sharesShortPriorMonth", None)

        short_float_pct = round(short_float * 100, 2) if short_float is not None else None
        short_ratio_days = round(short_ratio, 2) if short_ratio is not None else None
        
        short_mom_pct = None
        if shares_short and shares_short_prior and shares_short_prior > 0:
            short_mom_pct = round(((shares_short - shares_short_prior) / shares_short_prior) * 100, 2)

        squeeze_risk = "🟢 안정 (Low Risk)"
        if short_float_pct is not None and short_ratio_days is not None:
            if short_float_pct >= 20.0 and short_ratio_days >= 5.0:
                squeeze_risk = "🚨 숏스퀴즈 고위험 (High Squeeze Potential)"
            elif short_float_pct >= 10.0 and short_ratio_days >= 3.0:
                squeeze_risk = "⚠️ 숏스퀴즈 주의 (Moderate Potential)"
            elif short_float_pct >= 5.0:
                squeeze_risk = "💡 모니터링 구간 (Low-Moderate)"
        elif short_float_pct is not None:
            if short_float_pct >= 20.0:
                squeeze_risk = "🚨 숏스퀴즈 고위험 (High Squeeze Potential)"
            elif short_float_pct >= 10.0:
                squeeze_risk = "⚠️ 숏스퀴즈 주의 (Moderate Potential)"
            elif short_float_pct >= 5.0:
                squeeze_risk = "💡 모니터링 구간 (Low-Moderate)"

        intel["short_intel"] = {
            "short_percent_of_float": f"{short_float_pct:.2f}%" if short_float_pct is not None else "N/A",
            "short_ratio_days": f"{short_ratio_days:.2f}일" if short_ratio_days is not None else "N/A",
            "shares_short_formatted": f"{shares_short:,.0f}주" if shares_short else "N/A",
            "short_mom_change": f"{short_mom_pct:+.2f}%" if short_mom_pct is not None else "N/A",
            "squeeze_risk_level": squeeze_risk
        }
    except Exception:
        intel["short_intel"] = {
            "short_percent_of_float": "N/A",
            "short_ratio_days": "N/A",
            "shares_short_formatted": "N/A",
            "short_mom_change": "N/A",
            "squeeze_risk_level": "N/A"
        }
    return intel

def fetch_ownership_and_shorts(stock, info):
    data = {"insider_own": "N/A", "insider_trans": "N/A", "inst_own": "N/A", "inst_trans": "N/A"}
    try:
        ins_own_val = info.get("heldPercentInsiders", None)
        if ins_own_val is not None:
            data["insider_own"] = f"{ins_own_val * 100:.2f}%"
        inst_own_val = info.get("heldPercentInstitutions", None)
        if inst_own_val is not None:
            data["inst_own"] = f"{inst_own_val * 100:.2f}%"
    except Exception:
        pass
    return data

def fetch_earnings_calendar(stock, info, high_52_calc, low_52_calc):
    return {
        "earnings_date": "미정",
        "d_day": "",
        "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh", None) or high_52_calc,
        "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow", None) or low_52_calc
    }

def fetch_fundamentals_and_valuation(ticker: str, curr_price: float, high_52_calc, low_52_calc):
    stock = yf.Ticker(ticker)
    info, info_source = get_stock_info_with_retry(stock, retries=3)
    market_cap = info.get("marketCap", "N/A")
    trailing_pe = info.get("trailingPE", "N/A")
    forward_pe = info.get("forwardPE", "N/A")
    pbr = info.get("priceToBook", "N/A")
    ps_ratio = info.get("priceToSalesTrailing12Months", "N/A")
    roe_raw = info.get("returnOnEquity", None)
    roe_pct = round(roe_raw * 100, 2) if roe_raw is not None else "N/A"
    eps = info.get("trailingEps", None)
    forward_eps = info.get("forwardEps", None)
    bps = info.get("bookValue", None)
    target_mean_price = info.get("targetMeanPrice", "N/A")

    ownership_and_shorts = fetch_ownership_and_shorts(stock, info)
    hedge_and_short_intel = fetch_hedge_funds_and_short_intel(stock, info)
    earnings_cal = fetch_earnings_calendar(stock, info, high_52_calc, low_52_calc)

    return {
        "info_source": info_source,
        "market_cap_fmt": format_market_cap(market_cap),
        "trailing_pe": round(trailing_pe, 2) if isinstance(trailing_pe, (int, float)) else trailing_pe,
        "forward_pe": round(forward_pe, 2) if isinstance(forward_pe, (int, float)) else forward_pe,
        "pbr": round(pbr, 2) if isinstance(pbr, (int, float)) else pbr,
        "ps_ratio": round(ps_ratio, 2) if isinstance(ps_ratio, (int, float)) else ps_ratio,
        "roe": f"{roe_pct}%" if roe_pct != "N/A" else "N/A",
        "target_mean_price": target_mean_price,
        "ownership_and_shorts": ownership_and_shorts,
        "hedge_and_short_intel": hedge_and_short_intel,
        "earnings_calendar": earnings_cal,
        "value_models": {"graham": "산출불가", "peter_lynch": "산출불가", "roe_pbr": "산출불가"},
        "growth_models": {"forward_peg": "산출불가", "psr_target": "산출불가", "dcf_growth": "산출불가"}
    }

def fetch_sector_performance():
    sector_etfs = [
        ("XLK", "IT/기술"), ("XLC", "커뮤니케이션"), ("XLY", "임의소비재"),
        ("XLP", "필수소비재"), ("XLF", "금융"), ("XLV", "헬스케어"),
        ("XLI", "산업재"), ("XLE", "에너지"), ("XLB", "소재"),
        ("XLU", "유틸리티"), ("XLRE", "부동산")
    ]
    summary = {}
    for etf, name in sector_etfs:
        try:
            hist = yf.Ticker(etf).history(period="1mo")
            if len(hist) >= 2:
                pct_5d = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-5]) / hist['Close'].iloc[-5] * 100) if len(hist) >= 5 else 0.0
                pct_1m = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100)
                summary[etf] = {"sector_name": name, "return_5d": f"{pct_5d:+.2f}%", "return_1m": f"{pct_1m:+.2f}%", "latest_close": round(float(hist['Close'].iloc[-1]), 2)}
        except Exception:
            pass
    return summary

def fetch_news(ticker: str, limit: int = 5):
    try:
        stock = yf.Ticker(ticker)
        raw_news = stock.news
        if not raw_news:
            return []
        articles = []
        for n in raw_news[:limit]:
            content = n.get("content", {})
            title = content.get("title", "") if isinstance(content, dict) else n.get("title", "")
            if title:
                articles.append({"title": title, "publisher": "Yahoo Finance", "date": "최근", "link": f"https://finance.yahoo.com/quote/{ticker}"})
        return articles
    except Exception:
        return []

def fetch_macro_news(limit: int = 4):
    return [{"title": "Global Liquidity & Macro Update", "publisher": "MarketWatch", "date": "최근", "link": "https://finance.yahoo.com/quote/SPY"}]

def extract_clean_text(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return "\n".join([p["text"] if isinstance(p, dict) and "text" in p else str(p) for p in content])
    return str(content)

def summarize_user_strategy(raw_text: str) -> str:
    if not raw_text or raw_text == "분석 리포트 참조":
        return "분석 리포트 참조"
    text = raw_text.replace("\n", " ").strip()
    sentences = [s.strip() for s in re.split(r'[.!?]\s+', text) if len(s.strip()) > 5]
    summary = ". ".join(sentences[:2]) if sentences else text[:110]
    return summary[:120] + "..." if len(summary) > 120 else summary

def parse_full_trading_scenario(text):
    action = "홀딩"
    entry_grade = "분석 리포트 참조"
    entry_rr = "분석 리포트 참조"
    target_1 = "분석 리포트 참조"
    target_2 = ""
    sell_target = "분석 리포트 참조"
    buy_band = "분석 리포트 참조"
    stop_loss = "분석 리포트 참조"
    pyramiding = ""
    user_strategy_raw = ""

    match_action = re.search(r"\[최종\s*투자의견\s*[:\-]?\s*([^\]]+)\]", text)
    if match_action:
        op_text = match_action.group(1).strip()
        if "매수" in op_text and "관망" not in op_text:
            action = "매수"
        elif "매도" in op_text or "비중축소" in op_text:
            action = "매도"
        else:
            action = "홀딩"

    for line in text.split("\n"):
        line_clean = line.replace("*", "").replace("-", "").replace("•", "").strip()
        if "신규 진입 등급" in line_clean and entry_grade == "분석 리포트 참조":
            entry_grade = ":".join(line_clean.split(":")[1:]).strip()
        elif "예상 손익비" in line_clean:
            entry_rr = ":".join(line_clean.split(":")[1:]).strip()
        elif "사용자 대응 전략" in line_clean:
            user_strategy_raw = ":".join(line_clean.split(":")[1:]).strip()

    scenario_block = text
    if "[정밀 매매 시나리오]" in text:
        scenario_block = text.split("[정밀 매매 시나리오]")[1]

    for line in scenario_block.split("\n"):
        line_clean = line.replace("*", "").replace("-", "").replace("•", "").strip()
        if "1차 목표가" in line_clean:
            target_1 = ":".join(line_clean.split(":")[1:]).strip()
        elif "2차 목표가" in line_clean:
            target_2 = ":".join(line_clean.split(":")[1:]).strip()
        elif "매도가 밴드" in line_clean:
            sell_target = ":".join(line_clean.split(":")[1:]).strip()
        elif "분할 매수 밴드" in line_clean:
            buy_band = ":".join(line_clean.split(":")[1:]).strip()
        elif "손절" in line_clean:
            stop_loss = ":".join(line_clean.split(":")[1:]).strip()

    return action, entry_grade, entry_rr, target_1, target_2, sell_target, buy_band, stop_loss, pyramiding, summarize_user_strategy(user_strategy_raw)

# -------------------------------------------------------------
# 4. 사이드바 UI (통합 단일 입력 구조)
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ **AI Multi-Asset Analyst Pro**")
    
    MODEL_OPTIONS = {
        "Gemini 3.1 Flash Lite": "gemini-3.1-flash-lite",
        "Gemini 3.6 Flash Lite": "gemini-3.6-flash-lite",
        "Gemini 3.6 Flash": "gemini-3.6-flash"
    }
    
    selected_model_label = st.selectbox(
        "🤖 **AI 추론 모델 선택**",
        options=list(MODEL_OPTIONS.keys()),
        index=0
    )
    selected_model_id = MODEL_OPTIONS[selected_model_label]
    
    st.markdown("---")
    
    # 통합 티커 입력 (주식, GOLD, BTC 모두 입력 가능)
    raw_ticker_input = st.text_input("종목 또는 자산 입력 (예: TSLA, GOLD, BTC)", value=st.session_state.selected_ticker).strip()
    ticker_input, asset_display_name = normalize_ticker(raw_ticker_input)
    st.caption(f"🎯 인식된 자산: **{asset_display_name}** (`{ticker_input}`)")
    
    is_holding = st.checkbox("💼 **현재 보유 중인 종목/자산인가요?**", value=False)
    
    user_avg_price = 0.0
    user_shares = 0.0
    
    if is_holding:
        u_col1, u_col2 = st.columns(2)
        with u_col1:
            user_avg_price = st.number_input("내 평단가 ($)", min_value=0.0, value=0.0, step=0.5, format="%.2f")
        with u_col2:
            user_shares = st.number_input("보유 수량", min_value=0.0, value=0.0, step=1.0, format="%.1f")
            
    st.write("")
    analyze_btn = st.button("🚀 분석 & 백테스팅 실행", type="primary", use_container_width=True)
    st.divider()

    # 📌 종목별 트레이딩 히스토리 (주식, 금, 비트코인 통합 저장)
    st.markdown("#### 📌 **트레이딩 히스토리**")
    
    if st.session_state.history:
        tab_all, tab_buy, tab_sell, tab_hold = st.tabs(["전체", "🟢매수", "🔴매도", "🟡홀딩"])
        
        def render_history_card(t_code, data):
            action_badge = "🟢 매수" if data['action'] == "매수" else ("🔴 매도" if data['action'] == "매도" else "🟡 홀딩")
            with st.expander(f"**{data.get('asset_name', t_code)}** (${data['price']}) | {action_badge}", expanded=False):
                st.markdown(f"- **현재가:** `${data['price']}`")
                if data.get('my_avg', 0) > 0:
                    st.markdown(f"- **💼 내 평단:** `${data['my_avg']}` ({data.get('my_return', 'N/A')})")
                st.markdown(f"- **🎯 1차 목표가:** `{data.get('target_1', 'N/A')}`")
                st.markdown(f"- **📤 매도가 밴드:** `{data.get('sell_target', 'N/A')}`")
                st.markdown(f"- **📥 분할매수 밴드:** `{data.get('buy_band', 'N/A')}`")
                st.markdown(f"- **🛑 손절선:** `{data.get('stop_loss', 'N/A')}`")
                st.caption(f"분석 일시(KST): {data.get('time', 'N/A')}")

        with tab_all:
            for t_code, data in list(st.session_state.history.items())[::-1]:
                render_history_card(t_code, data)
        with tab_buy:
            for t_code, data in [item for item in list(st.session_state.history.items())[::-1] if item[1]['action'] == "매수"]:
                render_history_card(t_code, data)
        with tab_sell:
            for t_code, data in [item for item in list(st.session_state.history.items())[::-1] if item[1]['action'] == "매도"]:
                render_history_card(t_code, data)
        with tab_hold:
            for t_code, data in [item for item in list(st.session_state.history.items())[::-1] if item[1]['action'] == "홀딩"]:
                render_history_card(t_code, data)
                    
        st.write("")
        if st.button("🗑️ 히스토리 전체 삭제", use_container_width=True):
            st.session_state.history = {}
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.rerun()
    else:
        st.caption("분석을 실행하면 자산별 트레이딩 히스토리가 영구 저장됩니다.")

# -------------------------------------------------------------
# 5. 메인 분석 화면 & 분석 완료 즉시 렌더링 로직
# -------------------------------------------------------------
st.header(f"📊 [{asset_display_name}] 종합 밸류에이션 & 정밀 트레이딩 리포트")

is_macro_asset = ticker_input in ["GC=F", "BTC-USD"]

if analyze_btn:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            api_key = None
            
    with st.spinner(f"🔍 [{asset_display_name}] 기술적 지표, POC 매물대, VWAP 및 백테스팅 실행 중..."):
        tech_data, stock_date, fib_levels, high_52_calc, low_52_calc, raw_df, vol_profile = fetch_stock_technical_data(ticker_input)
        backtest_results = run_strategy_backtest(raw_df)
        options_data = fetch_nearest_options_data(ticker_input, retries=2)
        macro_data = fetch_macro_indicators()
        
        curr_p = tech_data.get('current_price', 0)
        fund_data = fetch_fundamentals_and_valuation(ticker_input, curr_p, high_52_calc, low_52_calc)
        sector_data = fetch_sector_performance()
        news_data = fetch_news(ticker_input, limit=5)
        macro_news_data = fetch_macro_news(limit=4)
        analyst_data = fetch_recent_upgrades_downgrades(ticker_input, months=2)
        
        info_source_flag = fund_data.get('info_source', 'stock.info')
        ownership = fund_data.get('ownership_and_shorts', {})
        hedge_short_intel = fund_data.get('hedge_and_short_intel', {})
        earnings_info = fund_data.get('earnings_calendar', {})
        
        my_return_str = "N/A"
        if is_holding and user_avg_price > 0 and user_shares > 0 and curr_p > 0:
            total_invested = user_avg_price * user_shares
            total_current = curr_p * user_shares
            pnl_dollar = total_current - total_invested
            pnl_pct = ((curr_p - user_avg_price) / user_avg_price) * 100
            my_return_str = f"{pnl_pct:+.2f}%"

        user_position_text = (
            f"사용자 보유 현황: 평단가 ${user_avg_price:.2f}, 보유수량 {user_shares:.1f}, 평가수익률 {my_return_str}"
            if is_holding and user_avg_price > 0 else "사용자 미보유 (신규 진입 검토 관점)"
        )

        if is_holding and user_avg_price > 0:
            strategy_instruction_text = f"""* **사용자 대응 전략**: [현재 사용자가 평단가 ${user_avg_price:.2f}, 평가수익률 {my_return_str}로 보유 중인 상태입니다. 반드시 '보유자 관점'의 전략만 단독 작성할 것. 1차/2차 목표가 도달 시 부분 익절 비중(예: 30% 매도) 및 손절선 이탈 시 전량 손절 계획을 명시할 것.]"""
        else:
            strategy_instruction_text = """* **사용자 대응 전략**: [현재 미보유 상태입니다. 반드시 '미보유자 신규 진입 관점'의 전략만 단독 작성할 것. 상단 [분할 매수 밴드]와 100% 일치하는 진입 가격대와 진입 비중을 명확히 제시할 것.]"""

        full_rag_payload = {
            "meta": {
                "asset_name": asset_display_name,
                "ticker": ticker_input,
                "is_macro_asset": is_macro_asset,
                "data_source": info_source_flag,
                "model_used": selected_model_label,
                "analysis_requested_at": get_current_kst_time_str(),
                "stock_data_date": stock_date
            },
            "technical_vwap_and_squeeze": tech_data,
            "volume_profile_poc_6m": vol_profile,
            "one_year_backtesting": backtest_results,
            "fibonacci_retracement_6m": fib_levels,
            "options_chain_nearest": options_data,
            "macro_indicators": macro_data,
            "fundamentals_or_macro_env": fund_data if not is_macro_asset else macro_data,
            "user_portfolio_status": user_position_text,
            "recent_news": news_data
        }

        full_json_str = json.dumps(full_rag_payload, indent=2, ensure_ascii=False)

        response_content = None
        if not api_key:
            response_content = "⚠️ GEMINI_API_KEY가 등록되지 않았습니다."
        else:
            if is_macro_asset:
                template = """
[매크로 헤지 자산 심층 분석 데이터 ({asset_name} / {ticker})]
1. 기술적 지표, VWAP, 볼린저 밴드 스퀴즈 및 6개월 최다 매물대(POC) (기준일: {stock_date}):
{tech_json}

2. 6개월 피보나치 되돌림 밴드:
{fib_json}

3. 최근 1년 백테스팅 결과:
{backtest_json}

4. 거시경제 지표 (미 10년물 실질금리, VIX, 달러인덱스):
{macro_json}

5. 사용자 보유 현황:
{user_position}

---
[지시사항]
위 데이터를 바탕으로 글로벌 매크로 헤지 자산 관점에서 정밀 리포트를 작성할 것:
1. **거시 유동성 환경 분석**: 미 실질금리(DGS10)와 달러인덱스(DXY), VIX 변동성이 {asset_name}에 미치는 영향.
2. **기술적 지표 및 POC 매물대 분석**: 현재가(${current_price})와 6개월 최다 매물대(POC), VWAP 간의 상관관계 및 지지/저항선 평가.
3. **[신규 진입 적격성 평가] 및 [정밀 매매 시나리오]**를 명확히 작성할 것 (가격 범위는 반드시 낮은 가격 ~ 높은 가격 오름차순 정렬).

[신규 진입 적격성 평가]
* **신규 진입 등급**: [...]
* **진입 적합성 분석**: [...]
* **예상 손익비 (Risk/Reward)**: [...]

[정밀 매매 시나리오]
* **분할 매수 밴드**: [...]
* **1차 목표가**: [...]
* **2차 목표가**: [...]
* **매도가 밴드**: [...]
* **손절(Stop-loss) 기준선**: [...]
* **불타기 조건**: [...]

[최종 투자의견: 적극매수 | 분할매수 | 홀딩(보유) | 비중축소 | 관망 중 택1]

{strategy_guide}
"""
                payload = {
                    "asset_name": asset_display_name,
                    "ticker": ticker_input,
                    "stock_date": stock_date,
                    "current_price": curr_p,
                    "tech_json": json.dumps(tech_data, indent=2, ensure_ascii=False),
                    "fib_json": json.dumps(fib_levels, indent=2, ensure_ascii=False),
                    "backtest_json": json.dumps(backtest_results, indent=2, ensure_ascii=False) if backtest_results else "데이터 부족",
                    "macro_json": json.dumps(macro_data, indent=2, ensure_ascii=False),
                    "user_position": user_position_text,
                    "strategy_guide": strategy_instruction_text
                }
            else:
                template = """
[주식 종목 심층 분석 데이터 ({ticker})]:
1. 기술적/수급, VWAP, 볼린저 밴드 스퀴즈 및 6개월 매물대 POC:
{tech_json}
2. 백테스팅 결과:
{backtest_json}
3. 피보나치 밴드:
{fib_json}
4. 펀더멘털 및 밸류에이션:
{fund_json}
5. 사용자 보유 현황 및 뉴스:
{user_position}

---
[지시사항]: 11개 섹터 분석, 숏스퀴즈 리스크 평가, 신규성 검증 및 16가지 체크리스트 반영, 오름차순 가격 밴드 정렬을 준수하여 정밀 리포트를 작성할 것.

[신규 진입 적격성 평가]
* **신규 진입 등급**: [...]
* **진입 적합성 분석**: [...]
* **예상 손익비 (Risk/Reward)**: [...]

[정밀 매매 시나리오]
* **분할 매수 밴드**: [...]
* **1차 목표가**: [...]
* **2차 목표가**: [...]
* **매도가 밴드**: [...]
* **손절(Stop-loss) 기준선**: [...]
* **불타기 조건**: [...]

[최종 투자의견: 적극매수 | 분할매수 | 홀딩(보유) | 비중축소 | 관망 중 택1]

{strategy_guide}
"""
                payload = {
                    "ticker": ticker_input,
                    "stock_date": stock_date,
                    "tech_json": json.dumps(tech_data, indent=2, ensure_ascii=False),
                    "backtest_json": json.dumps(backtest_results, indent=2, ensure_ascii=False) if backtest_results else "데이터 부족",
                    "fib_json": json.dumps(fib_levels, indent=2, ensure_ascii=False),
                    "fund_json": json.dumps(fund_data, indent=2, ensure_ascii=False),
                    "user_position": user_position_text,
                    "strategy_guide": strategy_instruction_text
                }

            prompt = PromptTemplate(input_variables=list(payload.keys()), template=template)
            llm = ChatGoogleGenerativeAI(model=selected_model_id, google_api_key=api_key)
            chain = prompt | llm
            
            try:
                res_ai = chain.invoke(payload)
                response_content = extract_clean_text(res_ai.content)
            except Exception as e:
                response_content = f"⚠️ 분석 생성 오류: {str(e)}"

        if response_content and not response_content.startswith("⚠️"):
            act, ent_grade, ent_rr, t1, t2, sell_b, buy_b, sl_b, pyr, u_strat_summary = parse_full_trading_scenario(response_content)
            st.session_state.history[ticker_input] = {
                "asset_name": asset_display_name,
                "action": act,
                "entry_grade": ent_grade,
                "entry_rr": ent_rr,
                "price": curr_p,
                "my_avg": user_avg_price if is_holding else 0,
                "my_return": my_return_str if is_holding else "미보유",
                "target_1": t1,
                "target_2": t2,
                "sell_target": sell_b,
                "buy_band": buy_b,
                "stop_loss": sl_b,
                "user_strategy": u_strat_summary,
                "time": get_current_kst_time_str()
            }
            save_history(st.session_state.history)

        st.session_state.last_analysis_result = {
            "ticker": ticker_input,
            "asset_name": asset_display_name,
            "is_macro_asset": is_macro_asset,
            "curr_p": curr_p,
            "is_holding": is_holding,
            "user_avg_price": user_avg_price,
            "user_shares": user_shares,
            "my_return_str": my_return_str,
            "tech_data": tech_data,
            "vol_profile": vol_profile,
            "backtest_results": backtest_results,
            "fib_levels": fib_levels,
            "macro_data": macro_data,
            "fund_data": fund_data,
            "response_content": response_content,
            "full_json_str": full_json_str
        }
        st.rerun()

# -------------------------------------------------------------
# 6. 결과 렌더링 파트 (주식 vs 금/비트코인 자동 분기)
# -------------------------------------------------------------
if st.session_state.last_analysis_result:
    res = st.session_state.last_analysis_result
    curr_p = res["curr_p"]
    tech_data = res["tech_data"]
    fib_levels = res["fib_levels"]
    backtest_results = res.get("backtest_results", None)
    fund_data = res["fund_data"]
    macro_data = res["macro_data"]
    is_macro = res["is_macro_asset"]
    resp_text = res.get("response_content", "")

    # 보유 포지션 분석
    if res["is_holding"] and res["user_avg_price"] > 0 and res["user_shares"] > 0 and curr_p > 0:
        total_invested = res["user_avg_price"] * res["user_shares"]
        total_current = curr_p * res["user_shares"]
        pnl_dollar = total_current - total_invested
        pnl_pct = ((curr_p - res["user_avg_price"]) / res["user_avg_price"]) * 100
        with st.container(border=True):
            st.markdown(f"#### 💼 **내 보유 포지션 분석 ({res['asset_name']})**")
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("내 매수 평단가", f"${res['user_avg_price']:,.2f}", f"{res['user_shares']:,.1f} 보유")
            p2.metric("총 매수 원금", f"${total_invested:,.2f}")
            p3.metric("현재 평가 금액", f"${total_current:,.2f}")
            p4.metric("평가 손익 (수익률)", f"${pnl_dollar:+,.2f}", f"{pnl_pct:+.2f}%")

    if is_macro:
        # [🪙 금 / 비트코인 전용 지표 카드]
        with st.container(border=True):
            st.markdown(f"**🪙 [{res['asset_name']}] 실시간 시세 및 매크로 헤지 지표**")
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("현재 시세", f"${curr_p:,.2f}")
            mc2.metric("미 10년물 국채금리", str(macro_data.get('us_10y_yield', {}).get('value', 'N/A')))
            mc3.metric("달러 인덱스 (DXY)", str(macro_data.get('dollar_index', {}).get('value', 'N/A')))
            mc4.metric("변동성 지수 (VIX)", str(macro_data.get('vix', {}).get('value', 'N/A')))
    else:
        # [🏢 주식 전용 메트릭 카드]
        with st.container(border=True):
            st.markdown(f"**🏢 [{res['asset_name']}] 핵심 시장 및 재무 지표**")
            r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4)
            r1_c1.metric("현재 주가", f"${curr_p}")
            r1_c2.metric("시가총액", str(fund_data.get('market_cap_fmt', 'N/A')))
            r1_c3.metric("PER (선행/후행)", f"{fund_data.get('forward_pe', 'N/A')} / {fund_data.get('trailing_pe', 'N/A')}")
            r1_c4.metric("PBR / PSR", f"{fund_data.get('pbr', 'N/A')} / {fund_data.get('ps_ratio', 'N/A')}")

    # 공통 기술적 지표 카드 (VWAP, POC 매물대)
    with st.container(border=True):
        st.markdown("##### 🧪 **퀀트 모멘텀, 스마트머니 VWAP & 6개월 최다 매물대 (POC)**")
        q_c1, q_c2, q_c3, q_c4 = st.columns(4)
        vwap_1y = tech_data.get('vwap_1y', 'N/A')
        q_c1.metric("1Y 누적 VWAP (장기 평단)", f"${vwap_1y}")
        vwap_20d = tech_data.get('vwap_20d', 'N/A')
        q_c2.metric("20일 단기 VWAP", f"${vwap_20d}")
        poc_val = tech_data.get('poc_price_6m', 'N/A')
        q_c3.metric("6M 최다 매물대 (POC)", f"${poc_val}")
        q_c4.metric("볼린저 밴드폭", f"{tech_data.get('bb_width_pct', 'N/A')}%")
        st.markdown(f"**⚡ 변동성 국면:** `{tech_data.get('bb_squeeze_status', 'N/A')}` | **🧱 70% 핵심 매물대 밴드:** `{tech_data.get('value_area_range_6m', 'N/A')}`")

    # 백테스팅 성과 카드
    if backtest_results:
        with st.container(border=True):
            st.markdown("##### 🔬 **과거 1년 퀀트 전략 백테스팅 시뮬레이션**")
            bh_ret = backtest_results.get("benchmark_buy_and_hold", 0.0)
            st.caption(f"📌 **벤치마크 (단순 보유 Buy & Hold 1년 수익률):** `{bh_ret:+.2f}%`")
            bt1, bt2 = st.columns(2)
            with bt1:
                s1 = backtest_results.get("strategy_1_momentum_squeeze", {})
                st.markdown("**🚀 전략 A: 모멘텀 스퀴즈 돌파**")
                st.metric("총 누적 수익률", f"{s1.get('total_ret', 0):+.2f}%", f"승률: {s1.get('win_rate', 0)}%")
            with bt2:
                s2 = backtest_results.get("strategy_2_vwap_mean_reversion", {})
                st.markdown("**🔄 전략 B: 1Y VWAP + RSI 되돌림**")
                st.metric("총 누적 수익률", f"{s2.get('total_ret', 0):+.2f}%", f"승률: {s2.get('win_rate', 0)}%")

    # 피보나치 되돌림 카드
    with st.container(border=True):
        st.markdown(f"##### 📐 **최근 6개월 피보나치 되돌림 밴드** (최고: `${fib_levels.get('high_6m', 'N/A')}` / 최저: `${fib_levels.get('low_6m', 'N/A')}`)")
        fb1, fb2, fb3, fb4 = st.columns(4)
        fb1.metric("23.6% 되돌림", f"${fib_levels.get('fib_23.6%', 'N/A')}")
        fb2.metric("38.2% 되돌림", f"${fib_levels.get('fib_38.2%', 'N/A')}")
        fb3.metric("50.0% 하프라인", f"${fib_levels.get('fib_50.0%', 'N/A')}")
        fb4.metric("61.8% 되돌림", f"${fib_levels.get('fib_61.8%', 'N/A')}")

    # AI 종합 분석 브리핑
    with st.container(border=True):
        h1, h2 = st.columns([0.65, 0.35])
        with h1:
            st.markdown(f"### 📝 **[{res['asset_name']}] AI 종합 분석 브리핑**")
        with h2:
            st.download_button(
                label="📥 분석용 JSON 다운로드",
                data=res["full_json_str"],
                file_name=f"{res['ticker']}_analysis_{datetime.now(KST).strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True
            )
        st.markdown(re.sub(r'(?<!\\)\$', r'\$', resp_text))