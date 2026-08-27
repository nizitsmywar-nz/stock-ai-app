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
    page_title="AI Stock Valuation Dashboard Pro",
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
        }
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
# 1. RAG 데이터 수집 모듈 (기술적 지표, BB Squeeze, VWAP, Volume Profile POC 등)
# -------------------------------------------------------------
def get_stock_info_with_retry(stock, retries=3):
    for attempt in range(retries):
        try:
            info = stock.info
            if isinstance(info, dict) and len(info) > 10 and any(k in info for k in ['marketCap', 'trailingPE', 'forwardPE', 'trailingEps', 'bookValue', 'currentPrice', 'regularMarketPrice', 'previousClose']):
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

# -------------------------------------------------------------
# 📌 정밀 전략 백테스팅 모듈 (1Y 일봉 기반)
# -------------------------------------------------------------
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

# -------------------------------------------------------------
# 📌 옵션 체인 스마트머니 수급 수집기 (안전 가드 추가)
# -------------------------------------------------------------
def fetch_nearest_options_data(ticker: str, retries: int = 3):
    # 원자재 선물(GC=F)이나 암호화폐(BTC-USD)는 옵션 체인이 없거나 에러를 유발하므로 즉시 차단
    if "=F" in ticker or "-USD" in ticker:
        return None

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

# -------------------------------------------------------------
# 📌 헤지펀드 & 공매도 분석 (비주식 자산 안전 가드 추가)
# -------------------------------------------------------------
def fetch_hedge_funds_and_short_intel(ticker: str, stock, info):
    intel = {
        "top_holders": [],
        "short_intel": {}
    }
    
    if "=F" in ticker or "-USD" in ticker:
        intel["short_intel"] = {
            "short_percent_of_float": "해당 없음 (원자재/코인)",
            "short_ratio_days": "해당 없음",
            "shares_short_formatted": "해당 없음",
            "short_mom_change": "해당 없음",
            "squeeze_risk_level": "🟢 비주식 자산 (해당 없음)"
        }
        return intel

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

def fetch_ownership_and_shorts(ticker: str, stock, info):
    data = {
        "insider_own": "N/A",
        "insider_trans": "N/A",
        "inst_own": "N/A",
        "inst_trans": "N/A"
    }
    if "=F" in ticker or "-USD" in ticker:
        data["insider_own"] = "해당 없음"
        data["insider_trans"] = "해당 없음"
        data["inst_own"] = "해당 없음"
        data["inst_trans"] = "해당 없음"
        return data

    try:
        ins_own_val = info.get("heldPercentInsiders", None)
        if ins_own_val is not None:
            data["insider_own"] = f"{ins_own_val * 100:.2f}%"
            
        inst_own_val = info.get("heldPercentInstitutions", None)
        if inst_own_val is not None:
            data["inst_own"] = f"{inst_own_val * 100:.2f}%"
    except Exception:
        pass

    try:
        ins_df = stock.insider_transactions
        if ins_df is not None and not ins_df.empty and 'Shares' in ins_df.columns:
            recent_ins = ins_df.head(15)
            net_shares = recent_ins['Shares'].dropna().sum()
            shares_out = info.get("sharesOutstanding", None)
            if shares_out and shares_out > 0:
                trans_pct = (net_shares / shares_out) * 100
                data["insider_trans"] = f"{trans_pct:+.2f}%"
            else:
                data["insider_trans"] = f"{net_shares:+,.0f}주"
    except Exception:
        pass

    try:
        inst_df = stock.institutional_holders
        if inst_df is not None and not inst_df.empty and '% Out' in inst_df.columns:
            tot_pct = inst_df['% Out'].sum() * 100
            data["inst_trans"] = f"{tot_pct:.2f}% (Top10)"
        elif inst_df is not None and not inst_df.empty and 'Shares' in inst_df.columns:
            tot_shares = inst_df['Shares'].sum()
            data["inst_trans"] = f"{tot_shares:,.0f}주 (Top10)"
    except Exception:
        pass

    return data

def fetch_earnings_calendar(ticker: str, stock, info, high_52_calc, low_52_calc):
    earnings_date_str = "해당 없음 (원자재/코인)" if ("=F" in ticker or "-USD" in ticker) else "미정"
    d_day_str = ""
    
    if "=F" not in ticker and "-USD" not in ticker:
        try:
            cal = stock.calendar
            if cal is not None and isinstance(cal, dict) and 'Earnings Date' in cal:
                e_dates = cal['Earnings Date']
                if isinstance(e_dates, list) and e_dates:
                    e_date = pd.to_datetime(e_dates[0])
                    earnings_date_str = e_date.strftime("%Y-%m-%d")
                    now = datetime.now()
                    days_diff = (e_date - now).days
                    if days_diff >= 0:
                        d_day_str = f"D-{days_diff}일"
                    else:
                        d_day_str = f"최근 발표완료 ({abs(days_diff)}일 전)"
            elif cal is not None and isinstance(cal, pd.DataFrame) and not cal.empty:
                if 'Earnings Date' in cal.index:
                    val = cal.loc['Earnings Date'].iloc[0]
                    earnings_date_str = str(val)[:10]
        except Exception:
            pass

    high_52w = info.get("fiftyTwoWeekHigh", None) or high_52_calc
    low_52w = info.get("fiftyTwoWeekLow", None) or low_52_calc

    return {
        "earnings_date": earnings_date_str,
        "d_day": d_day_str,
        "fiftyTwoWeekHigh": high_52w,
        "fiftyTwoWeekLow": low_52w
    }

def fetch_fundamentals_and_valuation(ticker: str, curr_price: float, high_52_calc, low_52_calc):
    stock = yf.Ticker(ticker)
    info, info_source = get_stock_info_with_retry(stock, retries=3)

    fast_info = {}
    try:
        if hasattr(stock, 'fast_info') and stock.fast_info:
            fast_info = stock.fast_info
    except Exception:
        pass

    market_cap = info.get("marketCap", None)
    if not market_cap and fast_info:
        market_cap = getattr(fast_info, 'market_cap', None) or fast_info.get('market_cap', "N/A")

    is_commodity_or_crypto = ("=F" in ticker or "-USD" in ticker)

    trailing_pe = "해당 없음" if is_commodity_or_crypto else info.get("trailingPE", "N/A")
    forward_pe = "해당 없음" if is_commodity_or_crypto else info.get("forwardPE", "N/A")
    pbr = "해당 없음" if is_commodity_or_crypto else info.get("priceToBook", "N/A")
    ps_ratio = "해당 없음" if is_commodity_or_crypto else info.get("priceToSalesTrailing12Months", "N/A")
    
    roe_raw = info.get("returnOnEquity", None)
    roe_pct = "해당 없음" if is_commodity_or_crypto else (round(roe_raw * 100, 2) if roe_raw is not None else "N/A")
    eps = None if is_commodity_or_crypto else info.get("trailingEps", None)
    forward_eps = None if is_commodity_or_crypto else info.get("forwardEps", None)
    bps = None if is_commodity_or_crypto else info.get("bookValue", None)
    revenue_per_share = None if is_commodity_or_crypto else info.get("revenuePerShare", None)
    target_mean_price = "해당 없음" if is_commodity_or_crypto else info.get("targetMeanPrice", "N/A")

    ownership_and_shorts = fetch_ownership_and_shorts(ticker, stock, info)
    hedge_and_short_intel = fetch_hedge_funds_and_short_intel(ticker, stock, info)
    earnings_cal = fetch_earnings_calendar(ticker, stock, info, high_52_calc, low_52_calc)

    earnings_growth = info.get("earningsGrowth", None)
    if earnings_growth and earnings_growth > 0:
        est_growth = min(earnings_growth * 100, 35.0)
    else:
        est_growth = 15.0

    def _value_model_sanity(value, label):
        if is_commodity_or_crypto:
            return "산출불가 (원자재/코인)"
        try:
            if not isinstance(value, (int, float)):
                return "산출불가"
            if not curr_price or curr_price <= 0:
                return value
            deviation = abs(value - curr_price) / curr_price
            high_per = isinstance(trailing_pe, (int, float)) and trailing_pe >= 60.0
            if deviation > 0.6 and high_per:
                return f"산출불가 (고PER 성장주 - 자산가치 모델 부적합, PER {trailing_pe:.1f}배)"
            if deviation > 0.6:
                return f"산출불가 (모델 괴리율 과다: {deviation*100:.0f}%)"
            return value
        except Exception:
            return "산출불가"

    value_models = {}
    if is_commodity_or_crypto:
        value_models = {"graham": "해당 없음", "peter_lynch": "해당 없음", "roe_pbr": "해당 없음"}
    else:
        try:
            if eps and bps and eps > 0 and bps > 0:
                raw_graham = round(math.sqrt(22.5 * float(eps) * float(bps)), 2)
                value_models["graham"] = _value_model_sanity(raw_graham, "graham")
            else:
                value_models["graham"] = "산출불가"
        except Exception:
            value_models["graham"] = "산출불가"

        try:
            if eps and eps > 0 and roe_raw and roe_raw > 0:
                raw_lynch = round(float(eps) * min(float(roe_raw) * 100, 25.0), 2)
                value_models["peter_lynch"] = _value_model_sanity(raw_lynch, "peter_lynch")
            else:
                value_models["peter_lynch"] = "산출불가"
        except Exception:
            value_models["peter_lynch"] = "산출불가"

        try:
            if bps and bps > 0 and roe_raw and roe_raw > 0:
                raw_roe_pbr = round(float(bps) * (float(roe_raw) / 0.10), 2)
                value_models["roe_pbr"] = _value_model_sanity(raw_roe_pbr, "roe_pbr")
            else:
                value_models["roe_pbr"] = "산출불가"
        except Exception:
            value_models["roe_pbr"] = "산출불가"

    used_growth_fallback = not (earnings_growth and earnings_growth > 0)

    def _sanity_capped(value, label):
        if is_commodity_or_crypto:
            return "산출불가 (원자재/코인)"
        try:
            if not isinstance(value, (int, float)) or not curr_price or curr_price <= 0:
                return "산출불가"
            deviation = abs(value - curr_price) / curr_price
            if deviation > 0.6:
                return f"산출불가 (모델 괴리율 과다: {deviation*100:.0f}%)"
            if used_growth_fallback:
                return f"{value} (참고용·추정성장률 가정치)"
            return value
        except Exception:
            return "산출불가"

    growth_models = {}
    if is_commodity_or_crypto:
        growth_models = {"forward_peg": "해당 없음", "psr_target": "해당 없음", "dcf_growth": "해당 없음"}
    else:
        f_eps = forward_eps if forward_eps and forward_eps > 0 else eps
        try:
            if f_eps and f_eps > 0:
                raw_peg = round(float(f_eps) * (est_growth * 1.5), 2)
                growth_models["forward_peg"] = _sanity_capped(raw_peg, "forward_peg")
            else:
                growth_models["forward_peg"] = "산출불가"
        except Exception:
            growth_models["forward_peg"] = "산출불가"

        try:
            if revenue_per_share and revenue_per_share > 0:
                raw_psr = round(float(revenue_per_share) * 5.0, 2)
                growth_models["psr_target"] = _sanity_capped(raw_psr, "psr_target")
            else:
                growth_models["psr_target"] = "산출불가"
        except Exception:
            growth_models["psr_target"] = "산출불가"

        try:
            if f_eps and f_eps > 0:
                wacc = 0.09
                g_long = 0.025
                pv_sum = 0
                cur_cf = float(f_eps)
                for y in range(1, 6):
                    cur_cf *= (1 + est_growth / 100)
                    pv_sum += cur_cf / ((1 + wacc) ** y)
                terminal_val = (cur_cf * (1 + g_long)) / (wacc - g_long)
                pv_terminal = terminal_val / ((1 + wacc) ** 5)
                raw_dcf = round(pv_sum + pv_terminal, 2)
                growth_models["dcf_growth"] = _sanity_capped(raw_dcf, "dcf_growth")
            else:
                growth_models["dcf_growth"] = "산출불가"
        except Exception:
            growth_models["dcf_growth"] = "산출불가"

    return {
        "info_source": info_source,
        "market_cap_fmt": format_market_cap(market_cap),
        "trailing_pe": trailing_pe,
        "forward_pe": forward_pe,
        "pbr": pbr,
        "ps_ratio": ps_ratio,
        "roe": roe_pct,
        "target_mean_price": target_mean_price,
        "ownership_and_shorts": ownership_and_shorts,
        "hedge_and_short_intel": hedge_and_short_intel,
        "earnings_calendar": earnings_cal,
        "value_models": value_models,
        "growth_models": growth_models
    }

def fetch_sector_performance():
    sector_etfs = [
        ("XLK", "IT/기술 (Technology)"),
        ("XLC", "커뮤니케이션 (Communication Services)"),
        ("XLY", "임의소비재 (Consumer Discretionary)"),
        ("XLP", "필수소비재 (Consumer Staples)"),
        ("XLF", "금융 (Financials)"),
        ("XLV", "헬스케어 (Health Care)"),
        ("XLI", "산업재 (Industrials)"),
        ("XLE", "에너지 (Energy)"),
        ("XLB", "소재 (Materials)"),
        ("XLU", "유틸리티 (Utilities)"),
        ("XLRE", "부동산 (Real Estate)")
    ]
    summary = {}
    for etf, name in sector_etfs:
        try:
            hist = yf.Ticker(etf).history(period="1mo")
            if len(hist) >= 2:
                pct_5d = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-5]) / hist['Close'].iloc[-5] * 100) if len(hist) >= 5 else ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100)
                pct_1m = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100)
                summary[etf] = {
                    "sector_name": name,
                    "return_5d": f"{pct_5d:+.2f}%",
                    "return_1m": f"{pct_1m:+.2f}%",
                    "latest_close": round(float(hist['Close'].iloc[-1]), 2)
                }
            else:
                summary[etf] = {"sector_name": name, "return_5d": "N/A", "return_1m": "N/A", "latest_close": "N/A"}
        except Exception:
            summary[etf] = {"sector_name": name, "return_5d": "N/A", "return_1m": "N/A", "latest_close": "N/A"}
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
            if isinstance(content, dict) and content:
                title = content.get("title", "")
                summary = content.get("summary", "")
                publisher = content.get("provider", {}).get("displayName", "Yahoo Finance")
                click_url = content.get("clickThroughUrl", {})
                link = click_url.get("url", "") if isinstance(click_url, dict) else click_url
                if not link:
                    link = content.get("canonicalUrl", {}).get("url", "")
                pub_date = str(content.get("pubDate", "최근"))[:10]
            else:
                title = n.get("title", "")
                summary = ""
                publisher = n.get("publisher", "Yahoo Finance")
                link = n.get("link", "")
                pub_time = n.get("providerPublishTime", None)
                pub_date = datetime.fromtimestamp(pub_time).strftime("%Y-%m-%d") if pub_time else "최근"
            
            if title:
                articles.append({
                    "title": title,
                    "summary": summary,
                    "publisher": publisher,
                    "date": pub_date,
                    "link": link or f"https://finance.yahoo.com/quote/{ticker}"
                })
        return articles
    except Exception:
        return []

def fetch_macro_news(limit: int = 4):
    macro_articles = []
    for sym in ["SPY", "TLT"]:
        try:
            stock = yf.Ticker(sym)
            raw = stock.news
            if raw:
                for n in raw[:2]:
                    content = n.get("content", {})
                    if isinstance(content, dict) and content:
                        title = content.get("title", "")
                        summary = content.get("summary", "")
                        publisher = content.get("provider", {}).get("displayName", "MarketWatch")
                        click_url = content.get("clickThroughUrl", {})
                        link = click_url.get("url", "") if isinstance(click_url, dict) else click_url
                        if not link:
                            link = content.get("canonicalUrl", {}).get("url", "")
                        pub_date = str(content.get("pubDate", "최근"))[:10]
                    else:
                        title = n.get("title", "")
                        summary = ""
                        publisher = n.get("publisher", "MarketWatch")
                        link = n.get("link", "")
                        pub_time = n.get("providerPublishTime", None)
                        pub_date = datetime.fromtimestamp(pub_time).strftime("%Y-%m-%d") if pub_time else "최근"
                    
                    if title and not any(a["title"] == title for a in macro_articles):
                        macro_articles.append({
                            "title": title,
                            "summary": summary,
                            "publisher": publisher,
                            "date": pub_date,
                            "link": link or f"https://finance.yahoo.com/quote/{sym}"
                        })
        except Exception:
            pass
    return macro_articles[:limit]

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
    
    if "보유자:" in text or "미보유자:" in text:
        parts = []
        if "보유자:" in text:
            holder_part = text.split("보유자:")[1].split("미보유자:")[0].strip()
            first_sen = re.split(r'[.!?]\s+', holder_part)[0].strip()
            if first_sen:
                parts.append(f"보유: {first_sen}")
        if "미보유자:" in text:
            non_holder_part = text.split("미보유자:")[1].strip()
            first_sen = re.split(r'[.!?]\s+', non_holder_part)[0].strip()
            if first_sen:
                parts.append(f"신규: {first_sen}")
        if parts:
            res = " | ".join(parts)
            return res[:130] + "..." if len(res) > 130 else res

    sentences = [s.strip() for s in re.split(r'[.!?]\s+', text) if len(s.strip()) > 5]
    if not sentences:
        return text[:110]
        
    summary = ". ".join(sentences[:2])
    if len(summary) > 120:
        summary = summary[:120] + "..."
    elif not summary.endswith("."):
        summary += "."
    return summary

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
        if "매수" in op_text and "관망" not in op_text and "보유" not in op_text:
            action = "매수"
        elif "매도" in op_text or "비중축소" in op_text or "차익실현" in op_text:
            action = "매도"
        elif "홀딩" in op_text or "보유" in op_text or "관망" in op_text:
            action = "홀딩"
    else:
        for line in text.split("\n"):
            line_str = line.replace("*", "").replace("#", "").replace("-", "").replace("•", "").strip()
            if "최종 투자 의견" in line_str or "최종 투자의견" in line_str:
                if "적극매수" in line_str or "분할매수" in line_str:
                    action = "매수"
                elif "비중축소" in line_str or "매도" in line_str or "차익실현" in line_str:
                    action = "매도"
                elif "홀딩" in line_str or "보유" in line_str or "관망" in line_str:
                    action = "홀딩"
                break

    match_entry_grade = re.search(r"\[신규\s*진입\s*적격성\s*평가\s*[:\-]?\s*([^\]]+)\]", text)
    if match_entry_grade:
        entry_grade = match_entry_grade.group(1).strip()
    
    for line in text.split("\n"):
        line_clean = line.replace("*", "").replace("-", "").replace("•", "").strip()
        if ("신규 진입 등급" in line_clean or "진입 등급" in line_clean) and entry_grade == "분석 리포트 참조":
            if ":" in line_clean:
                entry_grade = ":".join(line_clean.split(":")[1:]).strip()
        elif "예상 손익비" in line_clean or "손익비" in line_clean:
            if ":" in line_clean:
                entry_rr = ":".join(line_clean.split(":")[1:]).strip()
        elif "사용자 대응 전략" in line_clean or "사용자대응전략" in line_clean:
            if ":" in line_clean:
                user_strategy_raw = ":".join(line_clean.split(":")[1:]).strip()
            else:
                user_strategy_raw = line_clean.replace("사용자 대응 전략", "").replace("사용자대응전략", "").strip(" -:\t")

    scenario_block = text
    if "[정밀 매매 시나리오]" in text:
        after_header = text.split("[정밀 매매 시나리오]")[1]
        if "[최종 투자의견" in after_header:
            scenario_block = after_header.split("[최종 투자의견")[0]
        else:
            scenario_block = after_header

    for line in scenario_block.split("\n"):
        line_clean = line.replace("*", "").replace("-", "").replace("•", "").strip()
        
        if "1차 목표가" in line_clean or "1차목표가" in line_clean:
            if ":" in line_clean:
                target_1 = ":".join(line_clean.split(":")[1:]).strip()
        elif "2차 목표가" in line_clean or "2차목표가" in line_clean:
            if ":" in line_clean:
                target_2 = ":".join(line_clean.split(":")[1:]).strip()
        elif ("목표가" in line_clean or "익절 라인" in line_clean or "익절/" in line_clean) and target_1 == "분석 리포트 참조":
            if ":" in line_clean:
                target_1 = ":".join(line_clean.split(":")[1:]).strip()
        elif "매도가 밴드" in line_clean or "비중축소(익절) 밴드" in line_clean or "비중축소 밴드" in line_clean or "매도가" in line_clean:
            if ":" in line_clean:
                sell_target = ":".join(line_clean.split(":")[1:]).strip()
        elif "분할 매수 밴드" in line_clean or "분할매수 밴드" in line_clean or "분할 매수" in line_clean:
            if ":" in line_clean:
                buy_band = ":".join(line_clean.split(":")[1:]).strip()
        elif "손절" in line_clean or "Stop-loss" in line_clean:
            if ":" in line_clean:
                stop_loss = ":".join(line_clean.split(":")[1:]).strip()
        elif "불타기 조건" in line_clean or "불타기" in line_clean or "추가 매수 조건" in line_clean:
            if ":" in line_clean:
                pyramiding = ":".join(line_clean.split(":")[1:]).strip()

    user_strategy_summary = summarize_user_strategy(user_strategy_raw)
    return action, entry_grade, entry_rr, target_1, target_2, sell_target, buy_band, stop_loss, pyramiding, user_strategy_summary

TIER_1_FIRMS = [
    "goldman", "morgan stanley", "jpmorgan", "jp morgan", "citi", "citigroup",
    "bank of america", "bofa", "merrill", "ubs", "barclays", "deutsche bank",
    "hsbc", "bernstein", "credit suisse", "bnp paribas"
]

TIER_2_FIRMS = [
    "wells fargo", "rbc", "mizuho", "jefferies", "piper sandler", "wedbush",
    "baird", "oppenheimer", "bmo", "stifel", "td cowen", "cowen", "wolfe",
    "keybanc", "raymond james", "canaccord", "evercore", "truist", "guggenheim",
    "btig", "da davidson", "needham", "mmpm", "loop capital", "roth mkm", "bernstein"
]

def classify_analyst_tier(firm_name: str):
    f_lower = firm_name.lower()
    if any(k in f_lower for k in TIER_1_FIRMS):
        return "🌟 Tier 1 (글로벌 탑티어)", 1
    elif any(k in f_lower for k in TIER_2_FIRMS):
        return "✨ Tier 2 (주요 전문리서치)", 2
    else:
        return "🔎 Tier 3 (독립/부티크)", 3

def fetch_recent_upgrades_downgrades(ticker: str, months: int = 2):
    if "=F" in ticker or "-USD" in ticker:
        return []
    try:
        stock = yf.Ticker(ticker)
        upgrades = stock.upgrades_downgrades
        if upgrades is None or upgrades.empty:
            return []
        
        cutoff_date = datetime.now() - timedelta(days=months * 30)
        
        if isinstance(upgrades.index, pd.DatetimeIndex):
            filtered = upgrades[upgrades.index >= cutoff_date.strftime("%Y-%m-%d")]
        elif 'Date' in upgrades.columns:
            upgrades['Date'] = pd.to_datetime(upgrades['Date'])
            filtered = upgrades[upgrades['Date'] >= cutoff_date]
        else:
            filtered = upgrades.head(8)
            
        records = []
        for idx, row in filtered.head(8).iterrows():
            date_str = idx.strftime("%Y-%m-%d") if isinstance(idx, pd.Timestamp) else str(idx)[:10]
            firm_name = str(row.get("Firm", "N/A")).strip()
            from_g = str(row.get("FromGrade", "")).strip()
            to_g = str(row.get("ToGrade", "")).strip()
            action_raw = str(row.get("Action", "N/A")).strip()
            
            if from_g and from_g.lower() != "nan" and from_g != to_g:
                grade_str = f"{from_g} ➡️ {to_g}"
            else:
                grade_str = to_g if to_g and to_g.lower() != "nan" else "N/A"

            target_val = row.get("currentPriceTarget", None) or row.get("priceTarget", None) or row.get("TargetPrice", None)
            target_str = f"${float(target_val):.2f}" if pd.notnull(target_val) and target_val != "" else "-"

            tier_badge, tier_num = classify_analyst_tier(firm_name)

            records.append({
                "date": date_str,
                "firm": firm_name,
                "tier": tier_badge,
                "tier_num": tier_num,
                "action": action_raw,
                "grade_change": grade_str,
                "target_price": target_str
            })
        return records
    except Exception:
        return []

# -------------------------------------------------------------
# 2. 사이드바 UI
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ **AI Stock Analyst Pro**")
    
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
    # 사용자가 입력한 티커를 그대로 반영 (GOLD는 Barrick Gold, 금 시세는 GC=F 직접 입력)
    ticker_input = st.text_input("종목/자산 티커 (예: TSLA, GOLD, GC=F, BTC-USD)", value=st.session_state.get("selected_ticker", "TSLA")).upper().strip()
    
    is_holding = st.checkbox("💼 **현재 보유 중인 자산인가요?**", value=False)
    
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

    st.markdown("#### 📌 **트레이딩 히스토리**")
    
    if st.session_state.history:
        tab_all, tab_buy, tab_sell, tab_hold = st.tabs(["전체", "🟢매수", "🔴매도", "🟡홀딩"])
        
        def render_history_card(t_code, data):
            action_badge = "🟢 매수" if data['action'] == "매수" else ("🔴 매도" if data['action'] == "매도" else "🟡 홀딩")
            with st.expander(f"**{t_code}** (${data['price']}) | {action_badge}", expanded=False):
                st.markdown(f"- **현재가:** `${data['price']}`")
                
                entry_grade_val = data.get('entry_grade', '')
                if entry_grade_val and entry_grade_val != "분석 리포트 참조":
                    st.markdown(f"- **🆕 신규 진입 판정:** `{entry_grade_val}`")
                
                if data.get('my_avg', 0) > 0:
                    st.markdown(f"- **💼 내 평단:** `${data['my_avg']}` ({data.get('my_return', 'N/A')})")
                
                t1 = data.get('target_1') or data.get('take_profit', '분석 리포트 참조')
                t2 = data.get('target_2', '')
                st.markdown(f"- **🎯 1차 목표가:** `{t1}`")
                if t2:
                    st.markdown(f"- **🎯 2차 목표가:** `{t2}`")
                    
                st.markdown(f"- **📤 매도가 밴드:** `{data.get('sell_target', '분석 리포트 참조')}`")
                st.markdown(f"- **📥 분할매수 밴드:** `{data.get('buy_band', '분석 리포트 참조')}`")
                st.markdown(f"- **🛑 손절선:** `{data.get('stop_loss', '분석 리포트 참조')}`")
                
                if data.get('pyramiding'):
                    st.markdown(f"- **🔥 불타기 조건:** `{data['pyramiding']}`")
                
                strat_text = data.get('user_strategy', '')
                if strat_text and strat_text != "분석 리포트 참조":
                    st.markdown(f"- **💡 대응 전략:** `{strat_text}`")
                else:
                    st.markdown(f"- **💡 대응 전략:** `분석 리포트 참조`")
                    
                st.caption(f"분석 일시(KST): {data.get('time', 'N/A')}")

        with tab_all:
            for t_code, data in list(st.session_state.history.items())[::-1]:
                render_history_card(t_code, data)
                
        with tab_buy:
            buy_items = [item for item in list(st.session_state.history.items())[::-1] if item[1]['action'] == "매수"]
            if buy_items:
                for t_code, data in buy_items:
                    render_history_card(t_code, data)
            else:
                st.caption("매수 판정 내역이 없습니다.")

        with tab_sell:
            sell_items = [item for item in list(st.session_state.history.items())[::-1] if item[1]['action'] == "매도"]
            if sell_items:
                for t_code, data in sell_items:
                    render_history_card(t_code, data)
            else:
                st.caption("매도 판정 내역이 없습니다.")

        with tab_hold:
            hold_items = [item for item in list(st.session_state.history.items())[::-1] if item[1]['action'] == "홀딩"]
            if hold_items:
                for t_code, data in hold_items:
                    render_history_card(t_code, data)
            else:
                st.caption("홀딩 판정 내역이 없습니다.")
                    
        st.write("")
        if st.button("🗑️ 히스토리 전체 삭제", use_container_width=True):
            st.session_state.history = {}
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.rerun()
    else:
        st.caption("분석 내역이 여기에 영구 저장됩니다.")

# -------------------------------------------------------------
# 3. 메인 분석 화면 & 분석 완료 즉시 렌더링 로직
# -------------------------------------------------------------
st.header(f"📊 {ticker_input} 종합 밸류에이션 & 정밀 트레이딩 리포트")

if analyze_btn:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            api_key = None
            
    with st.spinner(f"🔍 [{ticker_input}] 기술적 지표, POC 매물대, 11개 섹터 수급 및 백테스팅 실행 중..."):
        tech_data, stock_date, fib_levels, high_52_calc, low_52_calc, raw_df, vol_profile = fetch_stock_technical_data(ticker_input)
        backtest_results = run_strategy_backtest(raw_df)
        options_data = fetch_nearest_options_data(ticker_input, retries=3)
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
            if is_holding and user_avg_price > 0 else "사용자 미보유 자산 (신규 진입 검토 관점)"
        )

        if is_holding and user_avg_price > 0:
            strategy_instruction_text = f"""* **사용자 대응 전략**: [현재 사용자가 평단가 ${user_avg_price:.2f}, 평가수익률 {my_return_str}로 자산을 보유 중인 상태입니다. 반드시 '보유자 관점'의 전략만 단독 작성할 것.]"""
        else:
            strategy_instruction_text = """* **사용자 대응 전략**: [현재 사용자가 자산을 보유하지 않은 '미보유 상태'입니다. 반드시 '미보유자 신규 진입 관점'의 전략만 단독 작성할 것.]"""

        full_rag_payload = {
            "meta": {
                "ticker": ticker_input,
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
            "ownership_and_short_interest": ownership,
            "hedge_funds_and_short_intel": hedge_short_intel,
            "earnings_calendar_and_52w": earnings_info,
            "macro_6_assets": macro_data,
            "global_macro_news": macro_news_data,
            "sector_performance_11_sectors": sector_data,
            "fundamentals_and_6_valuations": fund_data,
            "user_portfolio_status": user_position_text,
            "stock_recent_news": news_data,
            "analyst_upgrades_downgrades_2m": analyst_data
        }

        full_json_str = json.dumps(full_rag_payload, indent=2, ensure_ascii=False)

        response_content = None
        if not api_key:
            response_content = "⚠️ GEMINI_API_KEY가 등록되지 않았습니다."
        else:
            template = """
[RAG 심층 주입 데이터 ({ticker})]
1. 기술적/수급, VWAP, 볼린저 밴드 스퀴즈 및 6개월 매물대 POC:
{tech_json}

2. 최근 6개월 최다 매물대(POC) 및 70% 핵심 매물대(Value Area):
{poc_json}

3. 최근 1년 과거 데이터 기반 듀얼 전략 백테스팅 결과:
{backtest_json}

4. 최근 6개월 피보나치 되돌림 밴드:
{fib_json}

5. 옵션 체인 스마트머니 포지션:
{options_json}

6. 헤지펀드 지분 및 공매도 세력 분석:
{hedge_short_json}

7. 실적 발표 일정 및 52주 고저:
{earnings_json}

8. 매크로 및 6대 자산 실시간 지표:
{macro_json}

9. S&P 500 11개 전 섹터 실시간 등락률:
{sector_json}

10. 펀더멘털 및 밸류에이션:
{fund_json}

11. 사용자 보유 현황:
{user_position}

12. 최신 기사 및 애널리스트 투자의견:
{news_json}

위 데이터를 바탕으로 최고 수준의 퀀트 애널리스트 관점에서 정밀 리포트를 작성할 것:
- 거시경제 환경 및 11개 전 섹터 자금 순환매 분석.
- 기술적 지표, VWAP, POC 매물대 및 백테스팅 평가.
- 신규 진입 적격성 평가 및 구체적 매매 시나리오 (분할 매수 밴드, 목표가, 손절선) 제시.

[신규 진입 적격성 평가]
* **신규 진입 등급**: [적극 진입 추천 | 조정 시 분할 진입 | 돌파 확인 후 진입 | 진입 부적합(관망) 중 택1]
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
            prompt = PromptTemplate(
                input_variables=["ticker", "stock_date", "tech_json", "poc_json", "backtest_json", "fib_json", "options_json", "hedge_short_json", "earnings_json", "macro_json", "sector_json", "fund_json", "user_position", "strategy_guide", "news_json"],
                template=template
            )
            llm = ChatGoogleGenerativeAI(model=selected_model_id, google_api_key=api_key)
            chain = prompt | llm
            
            payload = {
                "ticker": ticker_input,
                "stock_date": stock_date,
                "tech_json": json.dumps(tech_data, indent=2, ensure_ascii=False),
                "poc_json": json.dumps(vol_profile, indent=2, ensure_ascii=False),
                "backtest_json": json.dumps(backtest_results, indent=2, ensure_ascii=False) if backtest_results else "백테스팅 데이터 부족",
                "fib_json": json.dumps(fib_levels, indent=2, ensure_ascii=False),
                "options_json": json.dumps(options_data, indent=2, ensure_ascii=False) if options_data else "옵션 데이터 없음 (원자재/코인)",
                "hedge_short_json": json.dumps(hedge_short_intel, indent=2, ensure_ascii=False),
                "earnings_json": json.dumps(earnings_info, indent=2, ensure_ascii=False),
                "macro_json": json.dumps(macro_data, indent=2, ensure_ascii=False),
                "sector_json": json.dumps(sector_data, indent=2, ensure_ascii=False),
                "fund_json": json.dumps(fund_data, indent=2, ensure_ascii=False),
                "user_position": user_position_text,
                "strategy_guide": strategy_instruction_text,
                "news_json": json.dumps(news_data, indent=2, ensure_ascii=False)
            }
            
            for delay in [0, 5, 10]:
                if delay > 0:
                    time.sleep(delay)
                try:
                    res = chain.invoke(payload)
                    response_content = extract_clean_text(res.content)
                    break
                except Exception:
                    continue
                    
            if not response_content:
                response_content = "⚠️ Gemini API 일시적 지연이 발생했습니다."

        if response_content and not response_content.startswith("⚠️"):
            act, ent_grade, ent_rr, t1, t2, sell_b, buy_b, sl_b, pyr, u_strat_summary = parse_full_trading_scenario(response_content)
            st.session_state.history[ticker_input] = {
                "action": act,
                "entry_grade": ent_grade,
                "entry_rr": ent_rr,
                "price": curr_p,
                "my_avg": user_avg_price if is_holding else 0,
                "my_return": my_return_str if is_holding else "미보유",
                "target_1": t1,
                "target_2": t2,
                "take_profit": t1,
                "sell_target": sell_b,
                "buy_band": buy_b,
                "stop_loss": sl_b,
                "pyramiding": pyr,
                "user_strategy": u_strat_summary,
                "time": get_current_kst_time_str()
            }
            save_history(st.session_state.history)

        st.session_state.last_analysis_result = {
            "ticker": ticker_input,
            "info_source": info_source_flag,
            "model_label": selected_model_label,
            "curr_p": curr_p,
            "is_holding": is_holding,
            "user_avg_price": user_avg_price,
            "user_shares": user_shares,
            "my_return_str": my_return_str,
            "tech_data": tech_data,
            "vol_profile": vol_profile,
            "backtest_results": backtest_results,
            "stock_date": stock_date,
            "fib_levels": fib_levels,
            "options_data": options_data,
            "macro_data": macro_data,
            "fund_data": fund_data,
            "sector_data": sector_data,
            "ownership": ownership,
            "hedge_short_intel": hedge_short_intel,
            "earnings_info": earnings_info,
            "response_content": response_content,
            "full_json_str": full_json_str,
            "macro_news_data": macro_news_data,
            "news_data": news_data,
            "analyst_data": analyst_data
        }
        st.rerun()

if st.session_state.last_analysis_result:
    res = st.session_state.last_analysis_result
    curr_p = res["curr_p"]
    ownership = res["ownership"]
    hedge_short_intel = res.get("hedge_short_intel", {})
    earnings_info = res["earnings_info"]
    tech_data = res["tech_data"]
    vol_profile = res.get("vol_profile", {})
    backtest_results = res.get("backtest_results", None)
    macro_data = res["macro_data"]
    fib_levels = res["fib_levels"]
    options_data = res["options_data"]
    fund_data = res["fund_data"]
    sector_data = res.get("sector_data", {})
    info_source = res.get("info_source", "stock.info")

    if info_source == "stock.info":
        st.markdown(f"📡 **데이터 소스:** `🟢 Yahoo Finance stock.info` ({res['ticker']})")
    else:
        st.markdown(f"📡 **데이터 소스:** `🟡 Yahoo Finance fast_info` ({res['ticker']})")

    if res["is_holding"] and res["user_avg_price"] > 0 and res["user_shares"] > 0 and curr_p > 0:
        total_invested = res["user_avg_price"] * res["user_shares"]
        total_current = curr_p * res["user_shares"]
        pnl_dollar = total_current - total_invested
        pnl_pct = ((curr_p - res["user_avg_price"]) / res["user_avg_price"]) * 100
        
        with st.container(border=True):
            st.markdown(f"#### 💼 **내 보유 포지션 분석 ({res['ticker']})**")
            p_c1, p_c2, p_c3, p_c4 = st.columns(4)
            p_c1.metric("내 매수 평단가", f"${res['user_avg_price']:,.2f}", f"{res['user_shares']:,.1f} 보유")
            p_c2.metric("총 투입 원금", f"${total_invested:,.2f}")
            p_c3.metric("현재 평가 금액", f"${total_current:,.2f}")
            p_c4.metric("평가 손익 (수익률)", f"${pnl_dollar:+,.2f}", f"{pnl_pct:+.2f}%")

    with st.container(border=True):
        st.markdown(f"**🏢 핵심 시장 지표 ({res['ticker']})**")
        r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4)
        r1_c1.metric("현재 가격", f"${curr_p}")
        r1_c2.metric("시가총액 / 거래대금", str(fund_data.get('market_cap_fmt', 'N/A')))
        r1_c3.metric("PER (선행/후행)", f"{fund_data.get('forward_pe', 'N/A')} / {fund_data.get('trailing_pe', 'N/A')}")
        r1_c4.metric("PBR / PSR", f"{fund_data.get('pbr', 'N/A')} / {fund_data.get('ps_ratio', 'N/A')}")
        
        st.divider()
        
        st.markdown("**👥 스마트머니 및 내부자 지분**")
        own_c1, own_c2, own_c3, own_c4 = st.columns(4)
        own_c1.metric("Insider Own", str(ownership.get('insider_own', 'N/A')))
        own_c2.metric("Insider Trans", str(ownership.get('insider_trans', 'N/A')))
        own_c3.metric("Inst Own", str(ownership.get('inst_own', 'N/A')))
        own_c4.metric("Inst Trans", str(ownership.get('inst_trans', 'N/A')))

        st.divider()

        st.markdown(f"**🐋 헤지펀드 & 수급 분석 ({res['ticker']})**")
        s_intel = hedge_short_intel.get("short_intel", {})
        
        sk1, sk2, sk3 = st.columns(3)
        sk1.metric("공매도 잔고 (Float)", s_intel.get("short_percent_of_float", "N/A"), f"MoM: {s_intel.get('short_mom_change', 'N/A')}")
        sk2.metric("상환 소요 일수 (DTC)", s_intel.get("short_ratio_days", "N/A"), "Days to Cover")
        sk3.metric("공매도 총 주수", s_intel.get("shares_short_formatted", "N/A"))
        
        st.markdown(f"**🎯 리스크 등급:** `{s_intel.get('squeeze_risk_level', 'N/A')}`")
        
        high_52 = earnings_info.get('fiftyTwoWeekHigh', 'N/A')
        diff_52h = round(((curr_p - high_52) / high_52) * 100, 1) if isinstance(high_52, (int, float)) and curr_p else None
        
        holders_list = hedge_short_intel.get("top_holders", [])
        if holders_list:
            df_holders = pd.DataFrame(holders_list)
            df_holders.columns = ["기관/펀드명", "보유 주식수", "지분율 (% Out)", "평가 가치"]
            st.dataframe(df_holders, use_container_width=True, hide_index=True)

        st.divider()

        st.markdown("**📅 주요 일정, 52주 가격 범위 & ATR/모멘텀**")
        s_c1, s_c2, s_c3, s_c4 = st.columns(4)
        s_c1.metric("실적/주요일정", str(earnings_info.get('earnings_date', 'N/A')), str(earnings_info.get('d_day', '')))
        
        low_52 = earnings_info.get('fiftyTwoWeekLow', 'N/A')
        s_c2.metric("52주 최고 / 최저가", f"${high_52} / ${low_52}", f"고점 대비 {diff_52h:+.1f}%" if diff_52h is not None else None)
        
        s_c3.metric("14일 ATR (일일 변동폭)", f"${tech_data.get('atr_14', 'N/A')}", f"2.0x 손절: ${tech_data.get('atr_stop_2_0x', 'N/A')}")
        s_c4.metric(
            "MACD (Signal)", 
            f"{tech_data.get('macd', 'N/A')} ({tech_data.get('macd_signal', 'N/A')})", 
            f"Hist: {tech_data.get('macd_hist', 'N/A'):+}" if isinstance(tech_data.get('macd_hist'), (int, float)) else None
        )

    if sector_data:
        with st.expander("🧭 **S&P 500 11개 전 섹터 실시간 등락 및 순환매 현황 [클릭하여 펼치기]**", expanded=False):
            s_rows = []
            for etf, s_info in sector_data.items():
                s_rows.append({
                    "티커": etf,
                    "섹터명": s_info.get("sector_name", ""),
                    "5일 등락률": s_info.get("return_5d", "N/A"),
                    "1개월 등락률": s_info.get("return_1m", "N/A"),
                    "현재가 ($)": f"${s_info.get('latest_close', 'N/A')}"
                })
            st.dataframe(pd.DataFrame(s_rows), use_container_width=True, hide_index=True)

    with st.container(border=True):
        st.markdown("##### 🧪 **퀀트 모멘텀, 스마트머니 VWAP & 6개월 최다 매물대 (POC)**")
        q_c1, q_c2, q_c3, q_c4 = st.columns(4)
        
        vwap_1y = tech_data.get('vwap_1y', 'N/A')
        diff_vwap1y = round(((curr_p - vwap_1y) / vwap_1y) * 100, 1) if isinstance(vwap_1y, (int, float)) and curr_p else None
        q_c1.metric("1Y 누적 VWAP", f"${vwap_1y}", f"현재가 {diff_vwap1y:+.1f}%" if diff_vwap1y is not None else None)

        vwap_20d = tech_data.get('vwap_20d', 'N/A')
        diff_vwap20 = round(((curr_p - vwap_20d) / vwap_20d) * 100, 1) if isinstance(vwap_20d, (int, float)) and curr_p else None
        q_c2.metric("20일 단기 VWAP", f"${vwap_20d}", f"현재가 {diff_vwap20:+.1f}%" if diff_vwap20 is not None else None)

        poc_val = tech_data.get('poc_price_6m', 'N/A')
        diff_poc = round(((poc_val - curr_p) / curr_p) * 100, 1) if isinstance(poc_val, (int, float)) and curr_p else None
        q_c3.metric("6M 최다 매물대 (POC)", f"${poc_val}", f"현재가 대비 {diff_poc:+.1f}%" if diff_poc is not None else "최대 거래 구간")

        bb_w = tech_data.get('bb_width_pct', 'N/A')
        q_c4.metric("볼린저 밴드폭", f"{bb_w}%" if bb_w != "N/A" else "N/A", "변동성 압축도")
        
        st.markdown(f"**⚡ 변동성 국면:** `{tech_data.get('bb_squeeze_status', 'N/A')}` | **🧱 70% 핵심 매물대:** `{tech_data.get('value_area_range_6m', 'N/A')}`")

    if backtest_results:
        with st.container(border=True):
            st.markdown("##### 🔬 **과거 1년 퀀트 전략 백테스팅 시뮬레이션**")
            bh_ret = backtest_results.get("benchmark_buy_and_hold", 0.0)
            st.caption(f"📌 **벤치마크 (Buy & Hold 1년 수익률):** `{bh_ret:+.2f}%`")
            
            bt_col1, bt_col2 = st.columns(2)
            with bt_col1:
                st.markdown("**🚀 전략 A: 모멘텀 스퀴즈 돌파**")
                s1 = backtest_results.get("strategy_1_momentum_squeeze", {})
                m1_1, m1_2, m1_3 = st.columns(3)
                m1_1.metric("총 누적 수익률", f"{s1.get('total_ret', 0):+.2f}%")
                m1_2.metric("승률", f"{s1.get('win_rate', 0)}%", f"총 {s1.get('trades_count', 0)}회")
                m1_3.metric("Profit Factor", f"{s1.get('profit_factor', 0)}")

            with bt_col2:
                st.markdown("**🔄 전략 B: 1Y VWAP + RSI 되돌림**")
                s2 = backtest_results.get("strategy_2_vwap_mean_reversion", {})
                m2_1, m2_2, m2_3 = st.columns(3)
                m2_1.metric("총 누적 수익률", f"{s2.get('total_ret', 0):+.2f}%")
                m2_2.metric("승률", f"{s2.get('win_rate', 0)}%", f"총 {s2.get('trades_count', 0)}회")
                m2_3.metric("Profit Factor", f"{s2.get('profit_factor', 0)}")

    with st.container(border=True):
        st.markdown(f"##### 📐 **최근 6개월 피보나치 되돌림 밴드** (최고: `${fib_levels.get('high_6m', 'N/A')}` / 최저: `${fib_levels.get('low_6m', 'N/A')}`)")
        fb1, fb2, fb3, fb4 = st.columns(4)
        f236, f382, f500, f618 = fib_levels.get('fib_23.6%', 'N/A'), fib_levels.get('fib_38.2%', 'N/A'), fib_levels.get('fib_50.0%', 'N/A'), fib_levels.get('fib_61.8%', 'N/A')
        
        fb1.metric("23.6% 되돌림", f"${f236}")
        fb2.metric("38.2% 되돌림", f"${f382}")
        fb3.metric("50.0% 하프라인", f"${f500}")
        fb4.metric("61.8% 되돌림", f"${f618}")

    with st.container(border=True):
        if options_data:
            exp_date = options_data['expiration_date']
            pc_rat = options_data['pc_volume_ratio']
            st.markdown(f"##### 🎯 **옵션 체인 스마트머니 포지션** `만기: {exp_date}` `P/C Ratio: {pc_rat}`")
            op_c1, op_c2, op_c3, op_c4 = st.columns(4)
            c_oi, c_vol, p_oi, p_vol = options_data['call_max_oi'], options_data['call_max_vol'], options_data['put_max_oi'], options_data['put_max_vol']
            op_c1.metric("콜옵션 Max OI", f"${c_oi['strike']}", f"OI: {c_oi['oi']:,}")
            op_c2.metric("콜옵션 Max Vol", f"${c_vol['strike']}", f"Vol: {c_vol['volume']:,}")
            op_c3.metric("풋옵션 Max OI", f"${p_oi['strike']}", f"OI: {p_oi['oi']:,}")
            op_c4.metric("풋옵션 Max Vol", f"${p_vol['strike']}", f"Vol: {p_vol['volume']:,}")
        else:
            st.markdown(f"##### 🎯 **옵션 체인 포지션**")
            st.info("해당 자산은 옵션 체인 거래가 지원되지 않습니다.")

    with st.container(border=True):
        head_col1, head_col2 = st.columns([0.65, 0.35])
        with head_col1:
            st.markdown(f"### 📝 **{res['model_label']} 종합 분석 브리핑**")
        with head_col2:
            st.download_button(
                label=f"📥 {res['ticker']} 분석용 JSON 다운로드",
                data=res["full_json_str"],
                file_name=f"{res['ticker']}_rag_analysis_data_{datetime.now(KST).strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True
            )
        clean_rendered_content = re.sub(r'(?<!\\)\$', r'\$', res["response_content"])
        st.markdown(clean_rendered_content)

    col_left, col_right = st.columns([0.9, 1.1])
    with col_left:
        with st.container(border=True):
            st.markdown(f"##### 📰 **{res['ticker']} 최신 뉴스 및 기사 원문**")
            if res.get("news_data"):
                for item in res.get("news_data", []):
                    st.markdown(f"**[{item['title']}]({item['link']})**")
                    if item.get("summary"):
                        st.markdown(f"> *{item['summary']}*")
                    st.caption(f"출처: {item['publisher']} | {item['date']}")
                    st.divider()
            else:
                st.info("수집된 최신 뉴스가 없습니다.")
                
    with col_right:
        with st.container(border=True):
            st.markdown(f"##### 🏛️ **{res['ticker']} 증권가 투자의견 변동**")
            if res.get("analyst_data"):
                df_analyst = pd.DataFrame(res.get("analyst_data", []))
                display_cols = ["date", "firm", "tier", "action", "grade_change", "target_price"]
                df_analyst = df_analyst[[c for c in display_cols if c in df_analyst.columns]]
                df_analyst.columns = ["일자", "증권사", "신뢰도", "구분", "의견변동", "목표가"]
                st.dataframe(df_analyst, use_container_width=True, hide_index=True)
            else:
                st.info("원자재 및 암호화폐 자산은 증권사 투자의견 리포트가 제공되지 않습니다.")